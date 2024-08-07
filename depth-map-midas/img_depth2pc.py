import cv2
import numpy as np
import torch
import time
import os
# from midas.dpt_depth import DPTDepthModel
# from midas.midas_net import MidasNet

#



# defining q matrix

Q = np.array(([[1, 0, 0, -160],
               [0, 1, 0, -120],
               [0, 0, 0, 350.0],
               [0, 0, 1/90, 0]]), dtype = np.float32)

print(Q)

#loading midas for depth estimation

model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

midas.eval()

#loading transforms to resize image and normalize it for midas

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform

elif model_type == "MiDaS_small":
    transform = midas_transforms.small_transform


path = "data/mono_calib_images/object_img/"

image_list = [filename for filename in os.listdir(path)]
print(image_list)

i = 0

for image in image_list:

    img = cv2.imread(path + image)

    start = time.time()

    image_filename = "output/mono_new/image" + str(i) + ".jpg"
    cv2.imwrite(image_filename, img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

      #applying input transforms

    input_batch = transform(img)

    while torch.no_grad():   #no traing for midas so no grad

        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1),
                                                     size=img.shape[:2],
                                                     mode="bicubic",
                                                     align_corners=False).squeeze()
        
        depth_map = prediction.detach().numpy()

        #applying minmax normalization to depth map

        depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX, dtype = cv2.CV_32F)

        #reprojecting depth map to 3d space

        points_3d = cv2.reprojectImageTo3D(depth_map, Q,  handleMissingValues = True)


        #getting rid of points with zero value-> not depth calculated

        mask_map = depth_map > 0.4

        #applying mask to points

        output_points = points_3d[mask_map]
        output_colors = img[mask_map]

        #saving output points and colors

        np.save("output/mono_new/points" + str(i) + ".npy", output_points)
        np.save("output/mono_new/colors" + str(i) + ".npy", output_colors)

        #saving point cloud

        end = time.time()
    
        print("time taken: ", end - start)

        fps = 1/(end - start)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        depth_map = (depth_map * 255).astype(np.uint8)
        depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

        #saving image and corresponding depth map in output folder
        depth_map_filename = "output/mono_new/depth_map" + str(i) + ".jpg"
        

        cv2.imwrite(depth_map_filename, depth_map)
        

        i += 1

        cv2.putText(img, f'FPS: {fps}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow("Image", img)
        cv2.imshow("Depth", depth_map)

        if cv2.waitKey(5) & 0xFF == ord('q'):
                    break


cv2.destroyAllWindows()