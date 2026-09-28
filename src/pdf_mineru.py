import os
import requests
import time
import zipfile
from pathlib import Path

# MinerU API 密钥：从环境变量 MINERU_API_KEY 读取（仅建库/PDF 解析时需要；纯问答不调用）。
# 申请：https://mineru.net  ；Windows 设置：setx MINERU_API_KEY "你的key"
api_key = os.environ.get("MINERU_API_KEY", "")

def get_task_id(file_name):
    url='https://mineru.net/api/v4/extract/task'
    header = {
        'Content-Type':'application/json',
        "Authorization":f"Bearer {api_key}".format(api_key)
    }
    pdf_url = 'https://vl-image.oss-cn-shanghai.aliyuncs.com/pdf/' + file_name
    data = {
        'url':pdf_url,
        'is_ocr':True,
        'enable_formula': False,
    }

    res = requests.post(url,headers=header,json=data)
    print(res.status_code)
    print(res.json())
    print(res.json()["data"])
    task_id = res.json()["data"]['task_id']
    return task_id

def get_result(task_id):
    url = f'https://mineru.net/api/v4/extract/task/{task_id}'
    header = {
        'Content-Type':'application/json',
        "Authorization":f"Bearer {api_key}".format(api_key)
    }

    while True:
        res = requests.get(url, headers=header)
        result = res.json()["data"]
        print(result)
        state = result.get('state')
        err_msg = result.get('err_msg', '')
        # 如果任务还在进行中，等待后重试
        if state in ['pending', 'running']:
            print("任务未完成，等待5秒后重试...")
            time.sleep(5)
            continue
        # 如果有错误，输出错误信息
        if err_msg:
            print(f"任务出错: {err_msg}")
            return
        # 如果任务完成，下载文件
        if state == 'done':
            full_zip_url = result.get('full_zip_url')
            if full_zip_url:
                local_filename = f"{task_id}.zip"
                print(f"开始下载: {full_zip_url}")
                r = requests.get(full_zip_url, stream=True)
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"下载完成，已保存到: {local_filename}")
                # 下载完成后自动解压
                unzip_file(local_filename)
            else:
                print("未找到 full_zip_url，无法下载。")
            return
        # 其他未知状态
        print(f"未知状态: {state}")
        return

# 解压zip文件的函数
def unzip_file(zip_path, extract_dir=None):
    """
    解压指定的zip文件到目标文件夹。
    :param zip_path: zip文件路径
    :param extract_dir: 解压目标文件夹，默认为zip同名目录
    """
    if extract_dir is None:
        extract_dir = zip_path.rstrip('.zip')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"已解压到: {extract_dir}")

# ============ 本地 PDF 上传解析（结果 zip 内含带页码的 content_list.json）============
def get_batch_upload_urls(file_names, model_version='vlm', enable_formula=True,
                          enable_table=True, language='ch', page_ranges=None):
    """为本地 PDF 申请批量上传地址，返回 (batch_id, file_urls)。
    model_version 默认 'vlm'（官方推荐，解析更鲁棒；旧的 pipeline 后端对部分 PDF 会报 -60017 retry limit）。
    数字版 PDF 无需强制 OCR，故 files 项不传 is_ocr（用其默认 false，由模型自动判断）。
    page_ranges 可选，形如 '1-3' 或 '2,4-6'，仅解析指定物理页（1-based），用于小范围诊断/分段。"""
    url = 'https://mineru.net/api/v4/file-urls/batch'
    header = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}
    files = []
    for f in file_names:
        item = {'name': Path(f).name}
        if page_ranges:
            item['page_ranges'] = page_ranges
        files.append(item)
    data = {
        'model_version': model_version,
        'enable_formula': enable_formula,
        'enable_table': enable_table,
        'language': language,
        'files': files,
    }
    res = requests.post(url, headers=header, json=data)
    res.raise_for_status()
    j = res.json()
    if j.get('code') != 0:
        raise RuntimeError(f"申请上传地址失败: {j}")
    return j['data']['batch_id'], j['data']['file_urls']


def upload_file(upload_url, pdf_path):
    """把本地 PDF 二进制 PUT 到 MinerU 给定的 OSS 预签名地址。
    注意：预签名 URL 已固化签名（v1），PUT 时不要额外带 Content-Type 头，否则 OSS 会返回 403；
    且文件名建议使用纯英文/数字，避免中文或【】：等特殊字符导致后端拉取失败。"""
    with open(pdf_path, 'rb') as f:
        data = f.read()
    r = requests.put(upload_url, data=data)
    print('上传字节数:', len(data), 'PUT 状态:', r.status_code)
    r.raise_for_status()


def poll_batch_result(batch_id, poll_interval=5, max_wait=1800):
    """轮询批量任务，直到每个文件都 done/failed，返回 extract_result 列表。"""
    url = f'https://mineru.net/api/v4/extract-results/batch/{batch_id}'
    header = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}
    waited = 0
    while True:
        res = requests.get(url, headers=header)
        res.raise_for_status()
        results = res.json()['data'].get('extract_result', [])
        states = [r.get('state') for r in results]
        print('批次状态:', states)
        if results and all(s in ('done', 'failed') for s in states):
            return results
        if waited >= max_wait:
            raise TimeoutError(f"批次 {batch_id} 超时未完成: {states}")
        time.sleep(poll_interval)
        waited += poll_interval


def download_and_unzip(zip_url, out_dir):
    """下载结果 zip 并解压到 out_dir，返回解压目录 Path。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / 'result.zip'
    with requests.get(zip_url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(out_dir)
    zip_path.unlink()
    return out_dir


def parse_local_pdf(pdf_path, out_dir, model_version='vlm', page_ranges=None):
    """完整流程：本地上传 -> 解析 -> 下载解压。返回解压目录 Path。
    page_ranges 可选（如 '1-3'），仅解析指定物理页。"""
    if not api_key:
        raise RuntimeError(
            "未设置 MinerU 密钥环境变量 MINERU_API_KEY（建库/PDF 解析必需）。"
            "请到 https://mineru.net 申请后 setx MINERU_API_KEY \"你的key\"，并重启终端。"
        )
    pdf_path = Path(pdf_path)
    batch_id, urls = get_batch_upload_urls([pdf_path.name], model_version=model_version,
                                           page_ranges=page_ranges)
    print('batch_id:', batch_id, 'page_ranges:', page_ranges)
    upload_file(urls[0], pdf_path)
    print('已上传:', pdf_path.name)
    results = poll_batch_result(batch_id)
    item = results[0]
    if item.get('state') != 'done':
        raise RuntimeError(f"解析失败: {item}")
    return download_and_unzip(item['full_zip_url'], out_dir)


if __name__ == "__main__":
    file_name = '【财报】中芯国际：中芯国际2024年年度报告.pdf'
    task_id = get_task_id(file_name)
    print('task_id:',task_id)
    get_result(task_id)
