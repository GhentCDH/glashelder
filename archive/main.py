import pixtral_inference
import qwen_inference
import torch
import os
import time
import cv2


def convert_tiff_to_png(folder_path):
    for image in os.listdir(folder_path):
        if image.endswith(".tiff") or image.endswith(".tif"):
            img = cv2.imread(os.path.join(folder_path, image))
            new_image_path = os.path.join(folder_path, image.replace('.tiff', '.png').replace('.tif', '.png'))
            cv2.imwrite(new_image_path, img)
            os.remove(os.path.join(folder_path, image))

def convert_jpg_to_png(folder_path):
    for image in os.listdir(folder_path):
        if image.endswith(".jpg") or image.endswith(".jpeg"):
            img = cv2.imread(os.path.join(folder_path, image))
            new_image_path = os.path.join(folder_path, image.replace('.jpg', '.png').replace('.jpeg', '.png'))
            cv2.imwrite(new_image_path, img)
            os.remove(os.path.join(folder_path, image))

def main(folder_path):

    # Parameters
    prompt = "These are images recorded by Belgians on a stereoscopic camera in China and annotated in French. Describe the image in detail, including any visible text. Provide four keywords that summarize the image,  and estimate the time period in which it was created."
    images = []
    pixtral_output = []
    qwen_output = []

    # Convert all images to PNG
    convert_jpg_to_png(folder_path)

    for image in os.listdir(folder_path):
        if image.endswith(".png"):
            images.append(os.path.join(folder_path, image))

    # Initialize the Qwen model
    qwen_start = time.time()
    qwen = qwen_inference.qwen_inference(prompt)
    for image in images:
        print(f'Now processing with Qwen {image.split("/")[-1]}...')
        output = str(qwen.run_inference(image, store_output=False, tokens=1028)).replace("\\n\\n", "\n")
        qwen_output.append((output, image))
        print(f'[Output Qwen]: {output}')
    
    del qwen
    qwen_stop = time.time()

    pixtral_start = time.time()
    # Initialize the Pixtral model
    pixtral = pixtral_inference.pixtral_inference()
    for image in images:
        print(f'Now processing with Pixtral {image.split("/")[-1]}...')
        output = pixtral.run_inference(image, prompt)
        pixtral_output.append((output, image))
        print(f'[Output Pixtral]: {output}')
    pixtral_stop = time.time()

    pixtral_time = pixtral_stop - pixtral_start
    qwen_time = qwen_stop - qwen_start
    total_time = pixtral_time + qwen_time

    with open(f"results/experiment_{str(folder_path).split('/')[-1]}_output.txt", "w") as f:
        f.write(f"Total time: {total_time}\n")
        f.write(f"Qwen time: {qwen_time}\n")
        f.write(f"Pixtral time: {pixtral_time}\n")
        f.write("\n")
        for idx, output in enumerate(pixtral_output):
            f.write("<><><><><><><><><><><><><><><>\n")
            f.write("IMAGE_NAME:" + output[1] + "\n")
            f.write("----PIXTRAL----\n")
            f.write(output[0] + "\n")
            f.write("<><><><><><><><><><><><><><><>\n")
            f.write("\n")
            f.write("----QWEN----\n")
            f.write(qwen_output[idx][0] + "\n")
            f.write("<><><><><><><><><><><><><><><>\n")
            f.write("\n")


main("dataset_images/ChinaStereo")



            

