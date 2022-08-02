# we actually do the work here >:)
# we actually do the work here >:)
import cv2
import time
import argparse
import os
import torch
import numpy as np
import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--notxt', action='store_true')
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--output_dir', type=str, default='./output')
args = parser.parse_args()

def main():
    model = posenet.load_model(args.model)
    model = model.cuda()
    output_stride = model.output_stride

    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    
    filenames = [
            f.path for f in os.scandir(args.image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg'))]
    

    photo = input("enter 1 to input image or 2 for webcam")
    # filenames = []
    if photo == 1:
        #take in image(s?)
        filenames = [
            f.path for f in os.scandir(args.image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg'))]
        input_image, draw_image, output_scale = posenet.read_imgfile(
            f, scale_factor=args.scale_factor, output_stride=output_stride)
    if photo == 2:
        #take photo with webcam and store image
        print("tbd")

    
    
    for f in filenames:
        input_image, draw_image, output_scale = posenet.read_imgfile(
            f, scale_factor=args.scale_factor, output_stride=output_stride)

        with torch.no_grad():
            input_image = torch.Tensor(input_image).cuda()

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(0),
                offsets_result.squeeze(0),
                displacement_fwd_result.squeeze(0),
                displacement_bwd_result.squeeze(0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.25)

        keypoint_coords *= output_scale

        return keypoint_coords, keypoint_scores
        # print("slouching?: ", calculate_if_slouching(keypoint_coords, keypoint_scores))

        """
        if args.output_dir:
            draw_image = posenet.draw_skel_and_kp(
                draw_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0, min_part_score=0.25)

            cv2.imwrite(os.path.join(args.output_dir, os.path.relpath(f, args.image_dir)), draw_image)

        if not args.notxt:
            print()
            print("Results for image: %s" % f)
            for pi in range(len(pose_scores)):
                if pose_scores[pi] == 0.:
                    break
                print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
                for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                    print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))
        """

   ## print('Average FPS:', len(filenames) / (time.time() - start))

    


def calculate_if_slouching(keypoint_coords, keypoint_scores):

    print("shape of key_coords: ", keypoint_coords.shape)
    print("shape of key_scores: ", keypoint_scores.shape)

    """
    take in the positional coordinates we need
    calculate angle
    assuming its waist and up, you would take left shoulder, left ear coordinates?
        calculate horizontal distance as difference between x coord of left shoulder and ear
        vertical distance is difference between their y coordinates
        take absolute value of vertical and horizontal distances
        arctan(vertical/horizontal distance) to find angle
    find what the angle is when youre not slouching
    if calculated angle is greater than not slouching angle, return True because slouching
    else False    
    """
    higher = 0.35 #TBD
    lower = 0.15
    

    keypoint_coords = keypoint_coords[:][:7][:] #error likely !!!!!!!!!!!!
    keypoint_scores = keypoint_scores[:][:7]

    print("shape of key_coords: ", keypoint_coords.shape)
    print("shape of key_scores: ", keypoint_scores.shape)
    
    for image in range(1, len(keypoint_coords)):
        
        head_sum = [0.0, 0.0]
        count = 1e-14

        print(keypoint_scores[image])

        for point in range(5):
            if keypoint_scores[image][point] > 0.01:
                head_sum += keypoint_coords[image][point]
                count += 1
        
        shd_sum = [0.0, 0.0]
        count2 = 1e-14

        for point in range(2):
            if keypoint_scores[image][point + 5] > 0.01:
                shd_sum += keypoint_coords[image][point + 5]
                count2 += 1
        
        print("count2", count2)

        head_avg = np.divide(head_sum, count)
        print("head_avg", head_avg)

        shd_avg = np.divide(shd_sum, count2)
        print("shd_avg", shd_avg)
        
        x_distance = abs(head_avg[0] - shd_avg[0])
        print("x_distance", x_distance)
        
        y_distance = abs(head_avg[1] - shd_avg[1])
        print("y_distance", y_distance)

        angle = np.arctan(y_distance/x_distance)
        print("angle", angle)
        print("\n")
        
        if angle < higher and angle > lower: #check if slouching
            print("good posture :)")
        else: 
            print("L posture >:(")
    
    
    

    #calculate the angle of the line between these two points
    """"if keypoint_coords[1, 3, 0] is not 0 and keypoint_coords[1, 5, 0] is not 0: #checks that left ear and shoulder coordinates arent 0
        x_distance = abs(keypoint_coords[1, 3, 0] - keypoint_coords[1, 5, 0])
        y_distance = abs(keypoint_coords[1, 3, 1] - keypoint_coords[1, 5, 1])
        angle = np.arctan(y_distance/x_distance)
        if angle > slouching_threshhold: #check if slouching
            return True 
        else: 
            return False
    

    if keypoint_coords[1, 4, 0] is not 0 and keypoint_coords[1, 6, 0] is not 0: #checks tharifhtt ear and shoulder coordinates arent 0
        x_distance = abs(keypoint_coords[1, 4, 0] - keypoint_coords[1, 6, 0])
        y_distance = abs(keypoint_coords[1, 4, 1] - keypoint_coords[1, 6, 1])  
        if angle > slouching_threshhold: #check if slouching
            return True 
        else:
            return False """""
    
    


if __name__ == "__main__":
    keyC, keyS = main()
    print("slouching?: ", calculate_if_slouching(keyC, keyS))
