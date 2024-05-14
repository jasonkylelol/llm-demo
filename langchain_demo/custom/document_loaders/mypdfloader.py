from typing import List
import json
import cv2
from PIL import Image
import numpy as np
import tqdm
import fitz # pyMuPDF里面的fitz包，不要与pip install fitz混淆

from pprint import pprint
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain_demo.custom.document_loaders.ocr import get_ocr

# PDF OCR 控制：只对宽高超过页面一定比例（图片宽/页面宽，图片高/页面高）的图片进行 OCR。
# 这样可以避免 PDF 中一些小图片的干扰，提高非扫描版 PDF 处理速度
PDF_OCR_THRESHOLD = (0.6, 0.6)

class RapidOCRPDFLoader(UnstructuredFileLoader):
    def _get_elements(self) -> List:
        def rotate_img(img, angle):
            '''
            img   --image
            angle --rotation angle
            return--rotated img
            '''
            
            h, w = img.shape[:2]
            rotate_center = (w/2, h/2)
            #获取旋转矩阵
            # 参数1为旋转中心点;
            # 参数2为旋转角度,正值-逆时针旋转;负值-顺时针旋转
            # 参数3为各向同性的比例因子,1.0原图，2.0变成原来的2倍，0.5变成原来的0.5倍
            M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
            #计算图像新边界
            new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
            new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
            #调整旋转矩阵以考虑平移
            M[0, 2] += (new_w - w) / 2
            M[1, 2] += (new_h - h) / 2

            rotated_img = cv2.warpAffine(img, M, (new_w, new_h))
            return rotated_img
        
        def handle_img_ocr(doc, page_index, page, resp):
            img_list = page.get_image_info(xrefs=True)
            if img_list:
                print(f"\nFound {len(img_list)} images on page: {page_index}")
            for img in img_list:
                ocr = get_ocr()
                # print(img)
                if xref := img.get("xref"):
                    bbox = img["bbox"]
                    # 检查图片尺寸是否超过设定的阈值
                    bbox_width = bbox[2] - bbox[0]
                    bbox_height = bbox[3] - bbox[1]
                    print(f"bbox width: {bbox_width:.2f} rect width: {page.rect.width:.2f} "
                        f"scale: {bbox_width / page.rect.width:.2f} threshold: {PDF_OCR_THRESHOLD[0]}")
                    print(f"bbox height: {bbox_height:.2f} rect height: {page.rect.height:.2f} "
                        f"scale: {(bbox_height) / (page.rect.height):.2f} threshold: {PDF_OCR_THRESHOLD[1]}")

                    if ((bbox_width) / (page.rect.width) < PDF_OCR_THRESHOLD[0]
                        and (bbox_height) / (page.rect.height) < PDF_OCR_THRESHOLD[1]):
                        continue
                    # if ((bbox[2] - bbox[0]) / (page.rect.width) < PDF_OCR_THRESHOLD[0]
                    #     or (bbox[3] - bbox[1]) / (page.rect.height) < PDF_OCR_THRESHOLD[1]):
                    #     continue
                    pix = fitz.Pixmap(doc, xref)
                    samples = pix.samples
                    if int(page.rotation)!=0:  #如果Page有旋转角度，则旋转图片
                        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)
                        tmp_img = Image.fromarray(img_array);
                        ori_img = cv2.cvtColor(np.array(tmp_img),cv2.COLOR_RGB2BGR)
                        rot_img = rotate_img(img=ori_img, angle=360-page.rotation)
                        img_array = cv2.cvtColor(rot_img, cv2.COLOR_RGB2BGR)
                    else:
                        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)

                    result, _ = ocr(img_array)
                    if result:
                        ocr_result = [line[1] for line in result]
                        ocr_result_text = "\n".join(ocr_result)
                        # print(f"ocr_result: {ocr_result_text}")
                        resp += f"{ocr_result_text}\n"
            return resp
        
        def handle_tables(page_index, page, resp):
            tabs = page.find_tables()
            if tabs.tables:
                print(f"\nFound {len(tabs.tables)} tables on page: {page_index}")
                for tab in tabs:
                    extracted_table = tab.extract()
                    # pprint(extracted_table)
                    # print(json.dumps(extracted_table, ensure_ascii=False))
            return resp

        def pdf2text(filepath):
            doc = fitz.open(filepath)
            resp = ""

            b_unit = tqdm.tqdm(total=doc.page_count, desc="RapidOCRPDFLoader context page index: 0")
            for page_index, page in enumerate(doc):
                b_unit.set_description("RapidOCRPDFLoader context page index: {}".format(page_index))
                b_unit.refresh()
                text = page.get_text("")
                resp += text + "\n"

                # if i != 0:
                #     print("\n---------------------------------------------------------------------")
                # print(text)

                resp = handle_img_ocr(doc, page_index, page, resp)

                resp = handle_tables(page_index, page, resp)

                # 更新进度
                b_unit.update(1)
            return resp

        text = pdf2text(self.file_path)
        from unstructured.partition.text import partition_text
        return partition_text(text=text, **self.unstructured_kwargs)


if __name__ == "__main__":
    loader = RapidOCRPDFLoader(file_path="/Users/tonysong/Desktop/test.pdf")
    docs = loader.load()
    print(docs)
