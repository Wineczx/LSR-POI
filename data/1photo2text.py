from PIL import Image
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import os
import json
import time
import logging
import argparse
logger = logging.getLogger()
# ... Configure logger ...

# Define the function
def image_to_text(skip_img_file, photo_dir_, write_filename_, cuda_no='cuda:0'):
    processor = AutoProcessor.from_pretrained("")
    model = Blip2ForConditionalGeneration.from_pretrained("", torch_dtype=torch.float16)
    device = torch.device(cuda_no if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    prompt = ''

    def image2text(image_filename):
        try:
            image = Image.open(image_filename).convert('RGB')
        except Image.UnidentifiedImageError:
            logger.error(f'转换失败, UnidentifiedImageError, 图片名称为:{image_filename}')
            return None
        except OSError as e:
            logger.error(f'转换失败, {repr(e)}, 图片名称为:{image_filename}')
            return None
        inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text

    # Load already processed record
    skip_img_set = set()
    if os.path.exists(skip_img_file):
        with open(skip_img_file, 'r') as f:
            skip_img_set = set(f.read().splitlines())

    images_dict = dict()
    for filename_ in os.listdir(photo_dir_):
        if filename_ in skip_img_set:
            continue
        
        parts = filename_.split('~')
        if len(parts) < 2:
            logger.error(f'文件名格式不正确，没有找到预期的"~": {filename_}')
            continue
        
        # Create a key based on the filename excluding the extension
        key = '~'.join(parts[:-1])
        photo_id = filename_
        images_dict.setdefault(key, []).append(filename_)

    total_num = len(images_dict)

    count = 1
    time_start = time.time()

    # Open the output file outside of the loop
    with open(write_filename_, 'w', encoding='utf-8') as f_output:
        for key, image_filename_list in images_dict.items():
            for image_filename in image_filename_list:
                image_text = image2text(os.path.join(photo_dir_, image_filename))
                if image_text:  # If conversion is successful
                    business_id, photo_id = key,image_filename.split('.')[0] # Assume photo_id does not contain '~'
                    image_json = {
                        "photo_id": photo_id,
                        "text": image_text,
                        "business_id": business_id
                    }
                    # Write the JSON object to the file, one record per line
                    f_output.write(json.dumps(image_json, ensure_ascii=False) + '\n')

                    # Update saved records
                    with open(skip_img_file, 'a') as f_skip:
                        f_skip.write(filename_ + '\n')

                # Logging the progress
                time_end = time.time()
                elapsed = time_end - time_start
                time_left = elapsed / count * (total_num - count) if count < total_num else 0
                time_left = time.strftime("%H:%M:%S", time.gmtime(time_left))
                print(f"{count}/{total_num}, time left: {time_left}, photo_id: {photo_id}")
                count += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', type=str, default='NYC', help='the region name of datasets(e.g. California)')
    parser.add_argument('--dataset_path', type=str, default='/NYC/', help='the index of the cuda')
    parser.add_argument('--cuda', type=str, default='1', help='the index of the cuda')
    args, _ = parser.parse_known_args()

    parent_path = os.path.join(args.dataset_path)

    # 初始化日志
    logfilename = 'transform_img_to_text.log'
    logfilepath = os.path.join(parent_path, logfilename)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    fh = logging.FileHandler(logfilepath, 'a', 'utf-8')
    fh.setLevel(logging.INFO)
    logging.basicConfig(
        level=logging.INFO,
        handlers = [sh, fh]
    )
    logger = logging.getLogger()

    # 图片转文字表述
    image_to_text(parent_path + 'skip_img_file.json', # 用于断点续传的跳过文件，每个地区一个
     parent_path + 'images/', # 已下载的meta-xxx.json的（每个POI的）图片集
     parent_path + 'image_description.json', # 输出文件
     'cuda:' + args.cuda) # 使用的GPU
