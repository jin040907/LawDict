from __future__ import annotations

from fastapi import Request

import openai # 혹은 사용 중인 LLM 라이브러리

import re

import json

import markdown

from fastapi import FastAPI, Request, Query

from fastapi.templating import Jinja2Templates

from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import pandas as pd

from sqlalchemy import create_engine, text

import uvicorn

# 1. 상단 임포트 구역
from pydantic import BaseModel

from duckduckgo_search import DDGS
import numpy as np
from typing import List, Dict, Tuple
import hashlib

app = FastAPI()

import os
from dotenv import load_dotenv

import openai

# ---------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    bill_name: str = "알 수 없는 법안"
    context: str = "내용 없음"
    conversation_history: list = []  # 대화 히스토리: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

# Load environment variables from .env file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

openai.api_key = os.getenv("OPENAI_API_KEY", "")

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Static 파일 서빙 설정
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# DB 연결 설정

DB_URL = "postgresql://cginside19:1234@localhost:5432/bill_db"

engine = create_engine(DB_URL)



def remove_report_titles(text: str) -> str:

    """

    보고서에 자동 추가된 제목들을 제거합니다.

    예: "**📋 주요 메타 데이터**", "**📊 예측 확률 분포**" 등

    """

    # 제목 패턴 제거 (이모지 + 제목 형식)

    patterns = [

        r'\*\*[📋📊🔍📈💡]\s*[^\*]+\*\*\s*\n',  # **📋 제목** 형식

        r'\*\*[^\*]+\*\*\s*\n',  # **제목** 형식 (이모지 없는 경우)

    ]

    result = text

    for pattern in patterns:

        result = re.sub(pattern, '', result, flags=re.MULTILINE)

    return result.strip()



def parse_prediction_table(md_text: str) -> list:

    """

    예측 확률 테이블을 파싱합니다.

    마크다운 테이블 형식에서 순위, 예측, 확률을 추출합니다.

    """

    predictions = []

   

    if not md_text:

        return predictions

   

    # "## 2. 모델 예측 결과" 섹션에서 테이블 찾기

    section_2_pattern = r'##\s+2\.\s+모델\s+예측\s+결과(.*?)(?=##\s+3\.|$)'

    section_2_match = re.search(section_2_pattern, md_text, re.DOTALL | re.IGNORECASE)

   

    if not section_2_match:

        return predictions

   

    section_2_content = section_2_match.group(1)

   

    # 테이블 헤더 찾기 (순위, 예측, 확률)

    table_header_pattern = r'\|\s*순위\s*\|\s*예측\s*\|\s*확률\s*\|'

    if not re.search(table_header_pattern, section_2_content):

        return predictions

   

    # 테이블 행 찾기 (| 숫자 | 텍스트 | 숫자.숫자 | 형식)

    table_row_pattern = r'\|\s*(\d+)\s*\|\s*([^\|]+)\s*\|\s*([\d.]+)\s*\|'

    matches = re.finditer(table_row_pattern, section_2_content)

   

    for match in matches:

        try:

            rank = int(match.group(1))

            prediction = match.group(2).strip()

            probability = float(match.group(3))

            predictions.append({

                'rank': rank,

                'prediction': prediction,

                'probability': probability

            })

        except (ValueError, IndexError):

            continue

   

    # 순위 순으로 정렬 (이미 순위대로 나올 가능성이 높지만 확실하게)

    predictions.sort(key=lambda x: x['rank'])

    # Top 3만 반환

    return predictions[:3]


def parse_social_impact_score(md_text: str):
    """report_md에서 사회적 영향력 점수(0~100) 추출. 0~100) 뒤의 숫자만 사용."""
    if not md_text:
        return None
    s = str(md_text)
    patterns = [
        r"사회적\s*영향력\s*점수\s*\(\s*0\s*~\s*100\s*\)\s*:?\s*\*?\*?\s*(\d+)",
        r"\(0\s*~\s*100\)\s*:?\s*\*?\*?\s*(\d+)",
        r"점수\s*\(\s*0\s*~\s*100\s*\)\s*:?\s*(\d+)",
    ]
    for p in patterns:
        m = re.search(p, s, re.IGNORECASE)
        if m:
            try:
                v = int(m.group(1))
                if 0 <= v <= 100:
                    return v
            except (ValueError, IndexError):
                pass
    return None


def split_expert_report(md_text: str) -> dict:

    """

    전문가 보고서를 파싱합니다. JSON 형식 또는 마크다운 형식을 자동 감지하여 처리합니다.

    원본 내용을 그대로 가져와서 마크다운→HTML 변환만 수행합니다.

    """

    keys = ["summary", "bill_info", "prediction", "evidence", "similar_cases", "social_impact", "next_action"]

   

    if not md_text or "준비 중" in md_text:

        return {k: "" for k in keys}



    txt = md_text.strip()

   

    # JSON 형식인지 확인 (첫 문자가 {)

    if txt.startswith('{'):

        try:

            data = json.loads(txt)

            result = {}

            # JSON의 원본 내용을 가져오되, 제목 제거 함수 적용

            for k in keys:

                if k in data and data[k]:

                    val = str(data[k]).strip()

                    # 보고서에 자동 추가된 제목 제거 (주요 메타 데이터, 예측 확률 분포 등)

                    val = remove_report_titles(val)

                    # social_impact 섹션에서 "분석:" 텍스트가 포함된 줄을 공백으로 변경
                    if k == 'social_impact':
                        lines = val.split('\n')
                        processed_lines = []
                        for i, line in enumerate(lines):
                            line_stripped = line.strip()
                            if re.match(r'^분석\s*:?\s*', line_stripped):
                                # "분석:" 또는 "분석 :" 또는 "분석"으로 시작하는 줄을 공백으로 변경
                                processed_lines.append('')
                            else:
                                processed_lines.append(line)
                        
                        val = '\n'.join(processed_lines)
                        
                        # "사회적 영향력 점수(0~100): 숫자" 뒤에 공백 개행 추가
                        val = re.sub(
                            r'(-\s*\*\*사회적 영향력 점수\(0~100\):\*\*\s*\d+[^\n]*)\n([^\n])',
                            r'\1\n\n\2',
                            val,
                            flags=re.MULTILINE
                        )
                        
                        # "참고 출처:" 앞에 공백 개행 추가
                        val = re.sub(
                            r'([^\n])\n(-\s*\*\*참고 출처:\*\*)',
                            r'\1\n\n\2',
                            val,
                            flags=re.MULTILINE
                        )

                    # evidence 섹션에서 "해석 요약:" 텍스트가 포함된 줄을 공백으로 변경
                    if k == 'evidence':
                        lines = val.split('\n')
                        processed_lines = []
                        for line in lines:
                            line_stripped = line.strip()
                            if re.match(r'^해석\s+요약\s*:?\s*', line_stripped):
                                # "해석 요약:" 또는 "해석 요약 :" 또는 "해석 요약"으로 시작하는 줄을 공백으로 변경
                                processed_lines.append('')
                            else:
                                processed_lines.append(line)
                        val = '\n'.join(processed_lines)
                        
                        # '[' 기호가 나오면서 제목이 나오는 부분 앞에 공백 개행 추가
                        # 줄 단위로 처리하여 마크다운 링크 문법 보호
                        lines = val.split('\n')
                        processed_lines_final = []
                        for i, line in enumerate(lines):
                            stripped = line.strip()
                            # '- **[제목]**:' 형태를 찾음 (리스트 항목의 제목)
                            # 마크다운 링크 '[텍스트](url)' 형태는 제외
                            if re.match(r'^-\s*\*\*\[[^\]]+\]\*\*?:', stripped) and not re.search(r'\[[^\]]+\]\(', stripped):
                                # 이전 줄이 비어있지 않으면 공백 추가
                                if processed_lines_final and processed_lines_final[-1].strip():
                                    processed_lines_final.append('')
                            processed_lines_final.append(line)
                        val = '\n'.join(processed_lines_final)

                    # next_action 섹션에서 '[' 기호 앞에 공백 개행 추가
                    if k == 'next_action':
                        # 더 확실한 방법: 줄 단위로 처리하여 이중 개행 추가
                        lines = val.split('\n')
                        processed_lines_final = []
                        for i, line in enumerate(lines):
                            stripped = line.strip()
                            # '- **[제목]**:' 또는 '- **[제목]**: ' 형태를 찾음 (리스트 항목의 제목)
                            # 마크다운 링크 '[텍스트](url)' 형태는 제외
                            if re.match(r'^-\s*\*\*\[[^\]]+\]\*\*?\s*:', stripped) and not re.search(r'\[[^\]]+\]\(', stripped):
                                # 이전 줄이 비어있지 않으면 공백 추가 (이중 개행)
                                if processed_lines_final and processed_lines_final[-1].strip():
                                    processed_lines_final.append('')
                                    processed_lines_final.append('')  # 이중 개행
                            processed_lines_final.append(line)
                        val = '\n'.join(processed_lines_final)
                        # 정규식으로도 한 번 더 처리하여 확실하게
                        val = re.sub(
                            r'([^\n])\n(-\s*\*\*\[[^\]]+\]\*\*?\s*:)',
                            r'\1\n\n\2',
                            val,
                            flags=re.MULTILINE
                        )

                    # bill_info 섹션에서 "요약:" 텍스트가 포함된 줄을 공백으로 변경
                    if k == 'bill_info':
                        lines = val.split('\n')
                        processed_lines = []
                        for line in lines:
                            line_stripped = line.strip()
                            if re.match(r'^요약\s*:?\s*', line_stripped):
                                # "요약:" 또는 "요약 :" 또는 "요약"으로 시작하는 줄을 공백으로 변경
                                processed_lines.append('')
                            else:
                                processed_lines.append(line)
                        val = '\n'.join(processed_lines)

                    # summary 섹션에서 "신뢰도" 포함 줄 완전히 제거 (마크다운 단계)
                    if k == 'summary':
                        lines = val.split('\n')
                        lines = [ln for ln in lines if not re.search(r'신뢰도\s*:', ln, re.IGNORECASE)]
                        val = '\n'.join(lines)

                    # similar_cases 섹션에서 표를 직접 HTML로 변환
                    if k == 'similar_cases':
                        # 먼저 표 앞에 공백 개행 2개 추가 (마크다운 파서가 표를 인식하도록)
                        lines = val.split('\n')
                        processed_lines = []
                        for i, line in enumerate(lines):
                            stripped = line.strip()
                            # 표 헤더 패턴: | 로 시작하고 끝남
                            if re.match(r'^\s*\|.*\|\s*$', stripped) and not re.match(r'^\s*\|[\s\-:|]+\|\s*$', stripped):
                                # 이전 줄 확인
                                prev_empty_count = 0
                                for j in range(i-1, -1, -1):
                                    if not lines[j].strip():
                                        prev_empty_count += 1
                                    else:
                                        break
                                
                                # 공백 개행이 2개 미만이면 추가
                                if prev_empty_count < 2:
                                    # 필요한 공백 개행 추가
                                    for _ in range(2 - prev_empty_count):
                                        processed_lines.append('')
                            
                            processed_lines.append(line)
                        val = '\n'.join(processed_lines)
                        
                        # 마크다운 표를 찾아서 HTML로 변환
                        def markdown_table_to_html(md_text):
                            lines = md_text.split('\n')
                            table_start = -1
                            separator_line = -1
                            table_end = -1
                            
                            # 표 시작, 구분선, 끝 찾기
                            for i, line in enumerate(lines):
                                stripped = line.strip()
                                # 표 헤더 패턴: | 로 시작하고 끝남 (구분선 제외)
                                if re.match(r'^\s*\|.*\|\s*$', stripped) and not re.match(r'^\s*\|[\s\-:|]+\|\s*$', stripped):
                                    if table_start == -1:
                                        table_start = i
                                    elif separator_line != -1:
                                        # 구분선 이후의 데이터 행
                                        table_end = i
                                # 구분선 패턴: | --- | 또는 |---|---|
                                elif re.match(r'^\s*\|[\s\-:|]+\|\s*$', stripped):
                                    if table_start != -1 and separator_line == -1:
                                        separator_line = i
                                elif table_start != -1 and separator_line != -1 and stripped and not stripped.startswith('|'):
                                    # 표가 끝남 (빈 줄이 아닌 다른 내용)
                                    table_end = i - 1
                                    break
                            
                            # 표 끝이 명시되지 않았으면 마지막 데이터 행까지
                            if table_start != -1 and separator_line != -1 and table_end == -1:
                                for i in range(separator_line + 1, len(lines)):
                                    stripped = lines[i].strip()
                                    if stripped and re.match(r'^\s*\|.*\|\s*$', stripped):
                                        table_end = i
                                    elif stripped and not stripped.startswith('|'):
                                        break
                            
                            if table_start != -1 and separator_line != -1 and table_end >= separator_line:
                                # 표를 HTML로 변환
                                html_table = ['<table>']
                                
                                # 헤더 처리
                                header_line = lines[table_start].strip()
                                # 헤더 셀 파싱 (첫 번째와 마지막 빈 문자열 제거)
                                header_cells_raw = header_line.split('|')
                                if header_cells_raw and not header_cells_raw[0].strip():
                                    header_cells_raw = header_cells_raw[1:]
                                if header_cells_raw and not header_cells_raw[-1].strip():
                                    header_cells_raw = header_cells_raw[:-1]
                                header_cells = [cell.strip() for cell in header_cells_raw if cell.strip()]
                                
                                # 구분선 정규화 (헤더 열 개수에 맞춤)
                                separator_line_content = lines[separator_line].strip()
                                normalized_separator = '|' + '|'.join(['---'] * len(header_cells)) + '|'
                                
                                if header_cells:
                                    html_table.append('<tr>')
                                    for cell in header_cells:
                                        html_table.append(f'<th>{cell}</th>')
                                    html_table.append('</tr>')
                                
                                # 데이터 행 처리
                                for i in range(separator_line + 1, table_end + 1):
                                    stripped = lines[i].strip()
                                    if not stripped or not re.match(r'^\s*\|.*\|\s*$', stripped):
                                        continue
                                    
                                    cells = [cell.strip() for cell in stripped.split('|')]
                                    # 첫 번째와 마지막이 빈 문자열일 수 있으므로 제거
                                    if cells and not cells[0].strip():
                                        cells = cells[1:]
                                    if cells and not cells[-1].strip():
                                        cells = cells[:-1]
                                    cells = [c.strip() for c in cells if c.strip()]
                                    
                                    # 헤더와 같은 개수의 셀만 사용 (구분선에 더 많은 열이 있어도 무시)
                                    if cells:
                                        html_table.append('<tr>')
                                        # 헤더 개수만큼만 사용, 부족하면 빈 셀 추가
                                        for idx in range(len(header_cells)):
                                            cell_content = cells[idx] if idx < len(cells) else ''
                                            html_table.append(f'<td>{cell_content}</td>')
                                        html_table.append('</tr>')
                                
                                html_table.append('</table>')
                                
                                # 원본 마크다운 표를 HTML로 교체
                                before_table = '\n'.join(lines[:table_start])
                                after_table = '\n'.join(lines[table_end+1:])
                                result = '\n\n'.join([before_table, '\n'.join(html_table), after_table]) if before_table.strip() or after_table.strip() else '\n'.join(html_table)
                                return result
                            
                            return md_text
                        
                        val = markdown_table_to_html(val)
                        
                        # HTML 테이블이 있으면 직접 처리
                        if '<table>' in val:
                            import re as regex_module
                            table_match = regex_module.search(r'<table>.*?</table>', val, regex_module.DOTALL)
                            if table_match:
                                table_html = table_match.group(0)
                                before_table = val[:table_match.start()].strip()
                                after_table = val[table_match.end():].strip()
                                
                                # 앞뒤 부분만 마크다운 변환
                                md = markdown.Markdown(extensions=['extra', 'nl2br', 'sane_lists', 'tables'])
                                before_html = md.convert(before_table) if before_table else ''
                                after_html = md.convert(after_table) if after_table else ''
                                
                                # HTML 테이블과 결합
                                html_content = before_html + table_html + after_html
                            else:
                                md = markdown.Markdown(extensions=['extra', 'nl2br', 'sane_lists', 'tables'])
                                html_content = md.convert(val)
                        else:
                            md = markdown.Markdown(extensions=['extra', 'nl2br', 'sane_lists', 'tables'])
                            html_content = md.convert(val)
                    else:
                        md = markdown.Markdown(extensions=['extra', 'nl2br', 'sane_lists', 'tables'])
                        html_content = md.convert(val)

                    # social_impact 섹션에서 HTML 변환 후에도 "분석:" 텍스트 제거 및 공백 추가
                    if k == 'social_impact':
                        # HTML에서 <p>분석:</p> 또는 <p>분석 :</p> 같은 패턴 제거
                        html_content = re.sub(r'<p>\s*분석\s*:?\s*</p>\s*', '', html_content, flags=re.IGNORECASE)
                        html_content = re.sub(r'<p>\s*분석\s*:?\s*<br\s*/?>\s*</p>\s*', '', html_content, flags=re.IGNORECASE)
                        # <strong>분석:</strong> 같은 패턴도 제거
                        html_content = re.sub(r'<strong>\s*분석\s*:?\s*</strong>\s*', '', html_content, flags=re.IGNORECASE)
                        # 일반 텍스트로 "분석:"이 포함된 줄 제거
                        html_content = re.sub(r'<p>\s*분석\s*:?\s*[^<]*</p>\s*', '<p></p>', html_content, flags=re.IGNORECASE)
                        
                        # "사회적 영향력 점수(0~100): 숫자" 뒤에 공백 추가
                        html_content = re.sub(
                            r'(<li[^>]*><strong[^>]*>사회적 영향력 점수\(0~100\):</strong>\s*\d+[^<]*</li>)\s*(<li[^>]*>)',
                            r'\1<p style="margin-top: 1em; margin-bottom: 0;"></p>\2',
                            html_content,
                            flags=re.IGNORECASE | re.DOTALL
                        )
                        # "참고 출처:" 앞에 공백 추가
                        html_content = re.sub(
                            r'(</li>)\s*(<li[^>]*>\s*(?:<p[^>]*>)?\s*<strong[^>]*>참고 출처:</strong>)',
                            r'\1<p style="margin-top: 1em; margin-bottom: 0;"></p>\2',
                            html_content,
                            flags=re.IGNORECASE | re.DOTALL
                        )
                        

                    # evidence 섹션에서 HTML 변환 후에도 "해석 요약:" 텍스트 제거
                    if k == 'evidence':
                        # HTML에서 <p>해석 요약:</p> 또는 <p>해석 요약 :</p> 같은 패턴 제거
                        html_content = re.sub(r'<p>\s*해석\s+요약\s*:?\s*</p>\s*', '', html_content, flags=re.IGNORECASE)
                        html_content = re.sub(r'<p>\s*해석\s+요약\s*:?\s*<br\s*/?>\s*</p>\s*', '', html_content, flags=re.IGNORECASE)
                        # <strong>해석 요약:</strong> 같은 패턴도 제거
                        html_content = re.sub(r'<strong>\s*해석\s+요약\s*:?\s*</strong>\s*', '', html_content, flags=re.IGNORECASE)
                        # 일반 텍스트로 "해석 요약:"이 포함된 줄 제거
                        html_content = re.sub(r'<p>\s*해석\s+요약\s*:?\s*[^<]*</p>\s*', '<p></p>', html_content, flags=re.IGNORECASE)

                    # bill_info 섹션에서 HTML 변환 후에도 "요약:" 텍스트 제거
                    if k == 'bill_info':
                        # HTML에서 <p>요약:</p> 또는 <p>요약 :</p> 같은 패턴 제거
                        html_content = re.sub(r'<p>\s*요약\s*:?\s*</p>\s*', '', html_content, flags=re.IGNORECASE)
                        html_content = re.sub(r'<p>\s*요약\s*:?\s*<br\s*/?>\s*</p>\s*', '', html_content, flags=re.IGNORECASE)
                        # <strong>요약:</strong> 같은 패턴도 제거
                        html_content = re.sub(r'<strong>\s*요약\s*:?\s*</strong>\s*', '', html_content, flags=re.IGNORECASE)
                        # 일반 텍스트로 "요약:"이 포함된 줄 제거
                        html_content = re.sub(r'<p>\s*요약\s*:?\s*[^<]*</p>\s*', '<p></p>', html_content, flags=re.IGNORECASE)

                    # next_action 섹션에서 HTML 변환 후에도 '[제목]**:' 앞에만 공백 추가
                    if k == 'next_action':
                        # 실제 HTML 구조: <li><strong>[제목]</strong>: </li> 또는 <li><p><strong>[제목]</strong>: </p></li> 형태
                        # </li> 뒤에 <li>가 오고 그 안에 <strong>[제목]</strong>: 패턴이 있는 경우 (중간에 <p> 태그가 있어도 됨)
                        html_content = re.sub(
                            r'(</li>)\s*(<li[^>]*>\s*(?:<p[^>]*>)?\s*<strong[^>]*>\[[^\]]+\]</strong>\s*:)',
                            r'\1<p style="margin-top: 1em; margin-bottom: 0;"></p>\2',
                            html_content,
                            flags=re.IGNORECASE | re.DOTALL
                        )
                        # </li> 뒤에 공백 없이 바로 <li>가 오는 경우
                        html_content = re.sub(
                            r'(</li>)(<li[^>]*>\s*(?:<p[^>]*>)?\s*<strong[^>]*>\[[^\]]+\]</strong>\s*:)',
                            r'\1<p style="margin-top: 1em; margin-bottom: 0;"></p>\2',
                            html_content,
                            flags=re.IGNORECASE | re.DOTALL
                        )
                        # </ul> 뒤에 <ul>이 오고 그 안에 <li><strong>[제목]</strong>: 형태가 있는 경우
                        html_content = re.sub(
                            r'(</ul>)\s*(<ul[^>]*>\s*<li[^>]*>\s*(?:<p[^>]*>)?\s*<strong[^>]*>\[[^\]]+\]</strong>\s*:)',
                            r'\1<p style="margin-top: 1em; margin-bottom: 0;"></p>\2',
                            html_content,
                            flags=re.IGNORECASE | re.DOTALL
                        )
                        # <p> 태그 안에 <strong>[제목]</strong>: 형태가 있는 경우 (이미 처리됨)
                        # 더 넓은 패턴: <strong>[제목]</strong>: 앞에 오는 모든 태그 뒤에 공백 추가
                        html_content = re.sub(
                            r'(</[^>]+>)\s*(<li[^>]*>\s*(?:<p[^>]*>)?\s*<strong[^>]*>\[[^\]]+\]</strong>\s*:)',
                            r'\1<p style="margin-top: 1em; margin-bottom: 0;"></p>\2',
                            html_content,
                            flags=re.IGNORECASE | re.DOTALL
                        )

                    # summary 섹션에서 "신뢰도:" 관련 내용 완전히 제거
                    if k == 'summary':
                        # 신뢰도가 포함된 줄 전체 제거 (다양한 HTML 패턴 고려)
                        # 패턴 1: <p>...</p> 태그 안에 신뢰도가 있는 경우
                        html_content = re.sub(
                            r'<p[^>]*>.*?신뢰도\s*:.*?</p>\s*',
                            '',
                            html_content,
                            flags=re.IGNORECASE | re.DOTALL
                        )
                        # 패턴 2: <li>...</li> 태그 안에 신뢰도가 있는 경우
                        html_content = re.sub(
                            r'<li[^>]*>.*?신뢰도\s*:.*?</li>\s*',
                            '',
                            html_content,
                            flags=re.IGNORECASE | re.DOTALL
                        )
                        # 패턴 3: <strong>신뢰도</strong> 형태
                        html_content = re.sub(
                            r'<strong[^>]*>.*?신뢰도\s*:.*?</strong>.*?',
                            '',
                            html_content,
                            flags=re.IGNORECASE | re.DOTALL
                        )
                        # 패턴 4: 일반 텍스트로 신뢰도가 포함된 줄
                        html_content = re.sub(
                            r'.*?신뢰도\s*:.*?(?:High|Medium|Low|high|medium|low).*?\n?',
                            '',
                            html_content,
                            flags=re.IGNORECASE | re.MULTILINE
                        )

                    result[k] = html_content

                else:

                    result[k] = ""

            return result

        except json.JSONDecodeError:

            # JSON 파싱 실패 시 마크다운으로 처리

            pass



    # 마크다운 형식 처리

    # ## 0. ~ ## 6. 형식의 섹션을 찾아서 분리

    section_pattern = r'##\s+(\d+)\.\s+([^\n]+)'

   

    # 모든 섹션 헤더 찾기

    matches = list(re.finditer(section_pattern, txt))

   

    result = {}

   

    for idx, match in enumerate(matches):

        section_num = int(match.group(1))
        
        # 섹션 번호를 키 인덱스로 매핑
        # 0->summary, 1->bill_info, 2->prediction, 3->evidence, 4->similar_cases, 5->social_impact, 6->next_action
        if section_num >= len(keys):
            continue  # 범위를 벗어난 섹션은 건너뜀
        
        key_index = section_num

        # 현재 섹션의 시작 위치 (헤더 다음 줄부터)
        start_pos = match.end()

        # 다음 섹션의 시작 위치 (없으면 끝까지)
        if idx + 1 < len(matches):
            end_pos = matches[idx + 1].start()
        else:
            end_pos = len(txt)

        # 섹션 내용 추출 (헤더 제목은 제외하고 내용만)
        section_body = txt[start_pos:end_pos].strip()
        
        # 디버깅: 섹션 5와 6 파싱 확인
        if section_num == 5 or section_num == 6:
            print(f"[DEBUG] 섹션 {section_num} 파싱 중...")
            print(f"  - 키 인덱스: {key_index}")
            print(f"  - 키 이름: {keys[key_index]}")
            print(f"  - 내용 길이: {len(section_body)}")
            print(f"  - 내용 일부 (처음 200자): {section_body[:200]}")

        # HTML로 변환
        if section_body:
            try:
                # 테이블이 리스트나 다른 블록 바로 다음에 오면 테이블 앞에 빈 줄 추가
                # 마크다운 테이블은 빈 줄로 구분되어야 함
                lines = section_body.split('\n')
                processed_lines = []

                for i, line in enumerate(lines):
                    line_stripped = line.strip()

                    # social_impact 섹션(섹션 5)에서 "분석:" 텍스트가 포함된 줄을 공백으로 변경
                    if section_num == 5 and key_index == 5:  # social_impact 섹션
                        if re.match(r'^분석\s*:?\s*', line_stripped):
                            # "분석:" 또는 "분석 :" 또는 "분석"으로 시작하는 줄을 공백으로 변경
                            processed_lines.append('')
                            continue

                    # evidence 섹션(섹션 3)에서 "해석 요약:" 텍스트가 포함된 줄을 공백으로 변경
                    if section_num == 3 and key_index == 3:  # evidence 섹션
                        if re.match(r'^해석\s+요약\s*:?\s*', line_stripped):
                            # "해석 요약:" 또는 "해석 요약 :" 또는 "해석 요약"으로 시작하는 줄을 공백으로 변경
                            processed_lines.append('')
                            continue
                        
                        # '[' 기호가 나오면서 제목이 나오는 부분 앞에 공백 개행 추가
                        if '[' in line and not line.strip().startswith('['):
                            # 이전 줄이 비어있지 않으면 공백 추가
                            if processed_lines and processed_lines[-1].strip():
                                processed_lines.append('')

                    # bill_info 섹션(섹션 1)에서 "요약:" 텍스트가 포함된 줄을 공백으로 변경
                    if section_num == 1 and key_index == 1:  # bill_info 섹션
                        if re.match(r'^요약\s*:?\s*', line_stripped):
                            # "요약:" 또는 "요약 :" 또는 "요약"으로 시작하는 줄을 공백으로 변경
                            processed_lines.append('')
                            continue

                    # summary 섹션(섹션 0)에서 "신뢰도" 포함 줄 완전히 제거
                    if section_num == 0 and key_index == 0:
                        if re.search(r'신뢰도\s*:', line_stripped, re.IGNORECASE):
                            continue

                    # 테이블 헤더 패턴 발견 (| 컬럼1 | 컬럼2 | 형식, --- 구분선이 아닌 경우)
                    if re.match(r'^\|\s*.+\s*\|', line_stripped) and not re.match(r'^\|\s*[-:]+', line_stripped):
                        # 이전 줄 확인
                        if i > 0:
                            prev_line = lines[i-1].strip()
                            # 이전 줄이 비어있지 않고 테이블이 아니면 빈 줄 추가
                            if prev_line and not prev_line.startswith('|'):
                                # processed_lines의 마지막 항목이 빈 줄이 아니면 추가
                                if not processed_lines or processed_lines[-1].strip():
                                    processed_lines.append('')  # 빈 줄 추가
                                    print(f"[DEBUG] 테이블 헤더 앞에 빈 줄 추가 (라인 {i+1})")

                    processed_lines.append(line)

                section_body_processed = '\n'.join(processed_lines)
                
                # social_impact 섹션(섹션 5)에서 "사회적 영향력 점수" 뒤와 "참고 출처" 앞에 공백 개행 추가
                if section_num == 5 and key_index == 5:  # social_impact 섹션
                    # "사회적 영향력 점수(0~100): 숫자" 뒤에 공백 개행 추가
                    section_body_processed = re.sub(
                        r'(-\s*\*\*사회적 영향력 점수\(0~100\):\*\*\s*\d+[^\n]*)\n([^\n])',
                        r'\1\n\n\2',
                        section_body_processed,
                        flags=re.MULTILINE
                    )
                    # "참고 출처:" 앞에 공백 개행 추가
                    section_body_processed = re.sub(
                        r'([^\n])\n(-\s*\*\*참고 출처:\*\*)',
                        r'\1\n\n\2',
                        section_body_processed,
                        flags=re.MULTILINE
                    )
                
                # evidence 섹션(섹션 3)에서 '[' 기호 앞에 공백 개행 추가
                if section_num == 3 and key_index == 3:  # evidence 섹션
                    # 줄 단위로 처리하여 마크다운 링크 문법 보호
                    lines = section_body_processed.split('\n')
                    processed_lines_final = []
                    for i, line in enumerate(lines):
                        stripped = line.strip()
                        # '- **[제목]**:' 형태를 찾음 (리스트 항목의 제목)
                        # 마크다운 링크 '[텍스트](url)' 형태는 제외
                        if re.match(r'^-\s*\*\*\[[^\]]+\]\*\*?:', stripped) and not re.search(r'\[[^\]]+\]\(', stripped):
                            # 이전 줄이 비어있지 않으면 공백 추가
                            if processed_lines_final and processed_lines_final[-1].strip():
                                processed_lines_final.append('')
                        processed_lines_final.append(line)
                    section_body_processed = '\n'.join(processed_lines_final)
                
                # next_action 섹션(섹션 6)에서 '[' 기호 앞에 공백 개행 추가
                if section_num == 6 and key_index == 6:  # next_action 섹션
                    # 더 확실한 방법: 줄 단위로 처리하여 이중 개행 추가
                    lines = section_body_processed.split('\n')
                    processed_lines_final = []
                    for i, line in enumerate(lines):
                        stripped = line.strip()
                        # '- **[제목]**:' 또는 '- **[제목]**: ' 형태를 찾음 (리스트 항목의 제목)
                        # 마크다운 링크 '[텍스트](url)' 형태는 제외
                        if re.match(r'^-\s*\*\*\[[^\]]+\]\*\*?\s*:', stripped) and not re.search(r'\[[^\]]+\]\(', stripped):
                            # 이전 줄이 비어있지 않으면 공백 추가 (이중 개행)
                            if processed_lines_final and processed_lines_final[-1].strip():
                                processed_lines_final.append('')
                                processed_lines_final.append('')  # 이중 개행
                        processed_lines_final.append(line)
                    section_body_processed = '\n'.join(processed_lines_final)
                    # 정규식으로도 한 번 더 처리하여 확실하게
                    section_body_processed = re.sub(
                        r'([^\n])\n(-\s*\*\*\[[^\]]+\]\*\*?\s*:)',
                        r'\1\n\n\2',
                        section_body_processed,
                        flags=re.MULTILINE
                    )
                
                # similar_cases 섹션(섹션 4)에서 표를 직접 HTML로 변환
                if section_num == 4 and key_index == 4:  # similar_cases 섹션
                    # 먼저 표 앞에 공백 개행 2개 추가 (마크다운 파서가 표를 인식하도록)
                    lines_for_table = section_body_processed.split('\n')
                    processed_lines_for_table = []
                    for i, line in enumerate(lines_for_table):
                        stripped = line.strip()
                        # 표 헤더 패턴: | 로 시작하고 끝남 (구분선 제외)
                        if re.match(r'^\s*\|.*\|\s*$', stripped) and not re.match(r'^\s*\|[\s\-:|]+\|\s*$', stripped):
                            # 이전 줄 확인
                            prev_empty_count = 0
                            for j in range(i-1, -1, -1):
                                if not lines_for_table[j].strip():
                                    prev_empty_count += 1
                                else:
                                    break
                            
                            # 공백 개행이 2개 미만이면 추가
                            if prev_empty_count < 2:
                                # 필요한 공백 개행 추가
                                for _ in range(2 - prev_empty_count):
                                    processed_lines_for_table.append('')
                        
                        processed_lines_for_table.append(line)
                    section_body_processed = '\n'.join(processed_lines_for_table)
                    
                    def markdown_table_to_html(md_text):
                        lines = md_text.split('\n')
                        table_start = -1
                        separator_line = -1
                        table_end = -1
                        
                        # 표 시작, 구분선, 끝 찾기
                        for i, line in enumerate(lines):
                            stripped = line.strip()
                            # 표 헤더 패턴: | 로 시작하고 끝남 (구분선 제외)
                            if re.match(r'^\s*\|.*\|\s*$', stripped) and not re.match(r'^\s*\|[\s\-:|]+\|\s*$', stripped):
                                if table_start == -1:
                                    table_start = i
                                elif separator_line != -1:
                                    # 구분선 이후의 데이터 행
                                    table_end = i
                            # 구분선 패턴: | --- | 또는 |---|---|
                            elif re.match(r'^\s*\|[\s\-:|]+\|\s*$', stripped):
                                if table_start != -1 and separator_line == -1:
                                    separator_line = i
                            elif table_start != -1 and separator_line != -1 and stripped and not stripped.startswith('|'):
                                # 표가 끝남 (빈 줄이 아닌 다른 내용)
                                table_end = i - 1
                                break
                        
                        # 표 끝이 명시되지 않았으면 마지막 데이터 행까지
                        if table_start != -1 and separator_line != -1 and table_end == -1:
                            for i in range(separator_line + 1, len(lines)):
                                stripped = lines[i].strip()
                                if stripped and re.match(r'^\s*\|.*\|\s*$', stripped):
                                    table_end = i
                                elif stripped and not stripped.startswith('|'):
                                    break
                        
                        if table_start != -1 and separator_line != -1 and table_end >= separator_line:
                            # 표를 HTML로 변환
                            html_table = ['<table>']
                            
                            # 헤더 처리
                            header_line = lines[table_start].strip()
                            # 헤더 셀 파싱 (첫 번째와 마지막 빈 문자열 제거)
                            header_cells_raw = header_line.split('|')
                            if header_cells_raw and not header_cells_raw[0].strip():
                                header_cells_raw = header_cells_raw[1:]
                            if header_cells_raw and not header_cells_raw[-1].strip():
                                header_cells_raw = header_cells_raw[:-1]
                            header_cells = [cell.strip() for cell in header_cells_raw if cell.strip()]
                            
                            # 구분선 정규화 (헤더 열 개수에 맞춤) - 실제로는 사용하지 않지만 참고용
                            separator_line_content = lines[separator_line].strip()
                            normalized_separator = '|' + '|'.join(['---'] * len(header_cells)) + '|'
                            
                            if header_cells:
                                html_table.append('<tr>')
                                for cell in header_cells:
                                    html_table.append(f'<th>{cell}</th>')
                                html_table.append('</tr>')
                            
                            # 데이터 행 처리
                            for i in range(separator_line + 1, table_end + 1):
                                stripped = lines[i].strip()
                                if not stripped or not re.match(r'^\s*\|.*\|\s*$', stripped):
                                    continue
                                
                                cells = [cell.strip() for cell in stripped.split('|')]
                                # 첫 번째와 마지막이 빈 문자열일 수 있으므로 제거
                                if cells and not cells[0].strip():
                                    cells = cells[1:]
                                if cells and not cells[-1].strip():
                                    cells = cells[:-1]
                                cells = [c.strip() for c in cells if c.strip()]
                                
                                # 헤더와 같은 개수의 셀만 사용 (구분선에 더 많은 열이 있어도 무시)
                                if cells:
                                    html_table.append('<tr>')
                                    # 헤더 개수만큼만 사용, 부족하면 빈 셀 추가
                                    for idx in range(len(header_cells)):
                                        cell_content = cells[idx] if idx < len(cells) else ''
                                        html_table.append(f'<td>{cell_content}</td>')
                                    html_table.append('</tr>')
                            
                            html_table.append('</table>')
                            
                            # 원본 마크다운 표를 HTML로 교체
                            before_table = '\n'.join(lines[:table_start])
                            after_table = '\n'.join(lines[table_end+1:])
                            result = '\n\n'.join([before_table, '\n'.join(html_table), after_table]) if before_table.strip() or after_table.strip() else '\n'.join(html_table)
                            return result
                        
                        return md_text
                    
                    section_body_processed = markdown_table_to_html(section_body_processed)
                    
                    # HTML 테이블이 있으면 직접 처리
                    if '<table>' in section_body_processed:
                        import re as regex_module
                        table_match = regex_module.search(r'<table>.*?</table>', section_body_processed, regex_module.DOTALL)
                        if table_match:
                            table_html = table_match.group(0)
                            before_table = section_body_processed[:table_match.start()].strip()
                            after_table = section_body_processed[table_match.end():].strip()
                            
                            # 앞뒤 부분만 마크다운 변환
                            md = markdown.Markdown(extensions=['extra', 'nl2br', 'sane_lists', 'tables'])
                            before_html = md.convert(before_table) if before_table else ''
                            after_html = md.convert(after_table) if after_table else ''
                            
                            # HTML 테이블과 결합
                            html_content = before_html + table_html + after_html
                        else:
                            md = markdown.Markdown(extensions=['extra', 'nl2br', 'sane_lists', 'tables'])
                            html_content = md.convert(section_body_processed)
                    else:
                        md = markdown.Markdown(extensions=['extra', 'nl2br', 'sane_lists', 'tables'])
                        html_content = md.convert(section_body_processed)
                else:
                    # 마크다운 확장 명시적으로 로드
                    md = markdown.Markdown(extensions=['extra', 'nl2br', 'sane_lists', 'tables'])
                    html_content = md.convert(section_body_processed)

                # social_impact 섹션에서 HTML 변환 후에도 "분석:" 텍스트 제거 및 공백 추가
                if section_num == 5 and key_index == 5:  # social_impact 섹션
                    # HTML에서 <p>분석:</p> 또는 <p>분석 :</p> 같은 패턴 제거
                    html_content = re.sub(r'<p>\s*분석\s*:?\s*</p>\s*', '', html_content, flags=re.IGNORECASE)
                    html_content = re.sub(r'<p>\s*분석\s*:?\s*<br\s*/?>\s*</p>\s*', '', html_content, flags=re.IGNORECASE)
                    # <strong>분석:</strong> 같은 패턴도 제거
                    html_content = re.sub(r'<strong>\s*분석\s*:?\s*</strong>\s*', '', html_content, flags=re.IGNORECASE)
                    # 일반 텍스트로 "분석:"이 포함된 줄 제거
                    html_content = re.sub(r'<p>\s*분석\s*:?\s*[^<]*</p>\s*', '<p></p>', html_content, flags=re.IGNORECASE)
                    
                    # "사회적 영향력 점수(0~100): 숫자" 뒤에 공백 추가
                    html_content = re.sub(
                        r'(<li[^>]*><strong[^>]*>사회적 영향력 점수\(0~100\):</strong>\s*\d+[^<]*</li>)\s*(<li[^>]*>)',
                        r'\1<p style="margin-top: 1em; margin-bottom: 0;"></p>\2',
                        html_content,
                        flags=re.IGNORECASE | re.DOTALL
                    )
                    # "참고 출처:" 앞에 공백 추가
                    html_content = re.sub(
                        r'(</li>)\s*(<li[^>]*>\s*(?:<p[^>]*>)?\s*<strong[^>]*>참고 출처:</strong>)',
                        r'\1<p style="margin-top: 1em; margin-bottom: 0;"></p>\2',
                        html_content,
                        flags=re.IGNORECASE | re.DOTALL
                    )

                # evidence 섹션에서 HTML 변환 후에도 "해석 요약:" 텍스트 제거
                if section_num == 3 and key_index == 3:  # evidence 섹션
                    # HTML에서 <p>해석 요약:</p> 또는 <p>해석 요약 :</p> 같은 패턴 제거
                    html_content = re.sub(r'<p>\s*해석\s+요약\s*:?\s*</p>\s*', '', html_content, flags=re.IGNORECASE)
                    html_content = re.sub(r'<p>\s*해석\s+요약\s*:?\s*<br\s*/?>\s*</p>\s*', '', html_content, flags=re.IGNORECASE)
                    # <strong>해석 요약:</strong> 같은 패턴도 제거
                    html_content = re.sub(r'<strong>\s*해석\s+요약\s*:?\s*</strong>\s*', '', html_content, flags=re.IGNORECASE)
                    # 일반 텍스트로 "해석 요약:"이 포함된 줄 제거
                    html_content = re.sub(r'<p>\s*해석\s+요약\s*:?\s*[^<]*</p>\s*', '<p></p>', html_content, flags=re.IGNORECASE)

                # next_action 섹션(섹션 6)에서 HTML 변환 후에도 '[제목]**:' 앞에만 공백 추가
                if section_num == 6 and key_index == 6:  # next_action 섹션
                    # 실제 HTML 구조: <li><strong>[제목]</strong>: </li> 또는 <li><p><strong>[제목]</strong>: </p></li> 형태
                    # </li> 뒤에 <li>가 오고 그 안에 <strong>[제목]</strong>: 패턴이 있는 경우 (중간에 <p> 태그가 있어도 됨)
                    html_content = re.sub(
                        r'(</li>)\s*(<li[^>]*>\s*(?:<p[^>]*>)?\s*<strong[^>]*>\[[^\]]+\]</strong>\s*:)',
                        r'\1<p style="margin-top: 1em; margin-bottom: 0;"></p>\2',
                        html_content,
                        flags=re.IGNORECASE | re.DOTALL
                    )
                    # </li> 뒤에 공백 없이 바로 <li>가 오는 경우
                    html_content = re.sub(
                        r'(</li>)(<li[^>]*>\s*(?:<p[^>]*>)?\s*<strong[^>]*>\[[^\]]+\]</strong>\s*:)',
                        r'\1<p style="margin-top: 1em; margin-bottom: 0;"></p>\2',
                        html_content,
                        flags=re.IGNORECASE | re.DOTALL
                    )
                    # </ul> 뒤에 <ul>이 오고 그 안에 <li><strong>[제목]</strong>: 형태가 있는 경우
                    html_content = re.sub(
                        r'(</ul>)\s*(<ul[^>]*>\s*<li[^>]*>\s*(?:<p[^>]*>)?\s*<strong[^>]*>\[[^\]]+\]</strong>\s*:)',
                        r'\1<p style="margin-top: 1em; margin-bottom: 0;"></p>\2',
                        html_content,
                        flags=re.IGNORECASE | re.DOTALL
                    )
                    # <p> 태그 안에 <strong>[제목]</strong>: 형태가 있는 경우 (이미 처리됨)
                    # 더 넓은 패턴: <strong>[제목]</strong>: 앞에 오는 모든 태그 뒤에 공백 추가
                    html_content = re.sub(
                        r'(</[^>]+>)\s*(<li[^>]*>\s*(?:<p[^>]*>)?\s*<strong[^>]*>\[[^\]]+\]</strong>\s*:)',
                        r'\1<p style="margin-top: 1em; margin-bottom: 0;"></p>\2',
                        html_content,
                        flags=re.IGNORECASE | re.DOTALL
                    )

                # bill_info 섹션에서 HTML 변환 후에도 "요약:" 텍스트 제거
                if section_num == 1 and key_index == 1:  # bill_info 섹션
                    # HTML에서 <p>요약:</p> 또는 <p>요약 :</p> 같은 패턴 제거
                    html_content = re.sub(r'<p>\s*요약\s*:?\s*</p>\s*', '', html_content, flags=re.IGNORECASE)
                    html_content = re.sub(r'<p>\s*요약\s*:?\s*<br\s*/?>\s*</p>\s*', '', html_content, flags=re.IGNORECASE)
                    # <strong>요약:</strong> 같은 패턴도 제거
                    html_content = re.sub(r'<strong>\s*요약\s*:?\s*</strong>\s*', '', html_content, flags=re.IGNORECASE)
                    # 일반 텍스트로 "요약:"이 포함된 줄 제거
                    html_content = re.sub(r'<p>\s*요약\s*:?\s*[^<]*</p>\s*', '<p></p>', html_content, flags=re.IGNORECASE)

                # summary 섹션에서 "신뢰도:" 관련 내용 완전히 제거
                if section_num == 0 and key_index == 0:  # summary 섹션
                    # 신뢰도가 포함된 줄 전체 제거 (다양한 HTML 패턴 고려)
                    # 패턴 1: <p>...</p> 태그 안에 신뢰도가 있는 경우
                    html_content = re.sub(
                        r'<p[^>]*>.*?신뢰도\s*:.*?</p>\s*',
                        '',
                        html_content,
                        flags=re.IGNORECASE | re.DOTALL
                    )
                    # 패턴 2: <li>...</li> 태그 안에 신뢰도가 있는 경우
                    html_content = re.sub(
                        r'<li[^>]*>.*?신뢰도\s*:.*?</li>\s*',
                        '',
                        html_content,
                        flags=re.IGNORECASE | re.DOTALL
                    )
                    # 패턴 3: <strong>신뢰도</strong> 형태
                    html_content = re.sub(
                        r'<strong[^>]*>.*?신뢰도\s*:.*?</strong>.*?',
                        '',
                        html_content,
                        flags=re.IGNORECASE | re.DOTALL
                    )
                    # 패턴 4: 일반 텍스트로 신뢰도가 포함된 줄
                    html_content = re.sub(
                        r'.*?신뢰도\s*:.*?(?:High|Medium|Low|high|medium|low).*?\n?',
                        '',
                        html_content,
                        flags=re.IGNORECASE | re.MULTILINE
                    )

                # 디버깅: 섹션 4 (similar_cases)의 경우 테이블 포함 여부 확인
                if section_num == 4:
                    # 테이블이 있는지 확인
                    has_table_marker = '|' in section_body
                    has_table_html = '<table' in html_content
                    
                    # 테이블 앞 빈 줄 확인
                    lines_before_table = []
                    for i, line in enumerate(processed_lines):
                        if re.match(r'^\|\s*.+\s*\|', line.strip()) and not re.match(r'^\|\s*[-:]+', line.strip()):
                            # 테이블 헤더 발견
                            if i > 0:
                                prev_line = processed_lines[i-1].strip() if i > 0 else ""
                                lines_before_table.append(f"라인 {i}: 이전 줄 = '{prev_line}' (빈 줄 여부: {not prev_line})")
                            break
                    
                    print(f"[DEBUG] 섹션 4 (유사 사례) 테이블 파싱 상태:")
                    print(f"  - 마크다운에 테이블 마커(|) 있음: {has_table_marker}")
                    print(f"  - HTML에 <table> 태그 있음: {has_table_html}")
                    print(f"  - 테이블 앞 빈 줄 정보: {lines_before_table}")
                    
                    if has_table_marker and not has_table_html:
                        print(f"[WARNING] 섹션 4에 테이블이 HTML로 변환되지 않음!")
                        print(f"[DEBUG] 원본 내용 일부 (처음 500자):\n{section_body[:500]}")
                        print(f"[DEBUG] 처리된 내용 일부 (처음 500자):\n{section_body_processed[:500]}")
                        print(f"[DEBUG] HTML 일부 (처음 500자):\n{html_content[:500]}")
                    elif has_table_html:
                        print(f"[DEBUG] 섹션 4 테이블 변환 성공!")

                result[keys[key_index]] = html_content

            except Exception as e:
                print(f"[ERROR] 마크다운 변환 오류 (섹션 {section_num}): {str(e)}")
                import traceback
                traceback.print_exc()
                result[keys[key_index]] = ""

        else:
            result[keys[key_index]] = ""

   

    # 없는 키는 빈 문자열로 채움

    for k in keys:

        if k not in result:

            result[k] = ""

           

    return result



@app.get("/api/search")
async def search_bills(
    request: Request, 
    q: str = Query("", description="검색어"),
    year: int = Query(None, description="연도"),
    month: int = Query(None, description="월"),
    day: int = Query(None, description="일")
):
    """
    법안 검색 API - 전체 데이터에서 검색
    """
    import sys
    print(f"[DEBUG] ========== Search API called ==========", file=sys.stderr, flush=True)
    print(f"[DEBUG] q='{q}' (type: {type(q).__name__}, len: {len(q) if q else 0})", file=sys.stderr, flush=True)
    print(f"[DEBUG] year={year} (type: {type(year).__name__ if year is not None else 'NoneType'})", file=sys.stderr, flush=True)
    print(f"[DEBUG] month={month} (type: {type(month).__name__ if month is not None else 'NoneType'})", file=sys.stderr, flush=True)
    print(f"[DEBUG] day={day} (type: {type(day).__name__ if day is not None else 'NoneType'})", file=sys.stderr, flush=True)
    print(f"[DEBUG] Request URL: {request.url}", file=sys.stderr, flush=True)
    print(f"[DEBUG] Query params dict: {dict(request.query_params)}", file=sys.stderr, flush=True)
    
    try:
        # 데이터베이스 연결 확인
        print(f"[DEBUG] Engine URL: {engine.url}", file=sys.stderr, flush=True)
        print(f"[DEBUG] Testing database connection...", file=sys.stderr, flush=True)
        
        with engine.connect() as conn:
            print(f"[DEBUG] Database connection successful", file=sys.stderr, flush=True)
            # 기본 쿼리 구성
            where_conditions = []
            params = {}
            
            # 검색어 조건 - PostgreSQL의 ILIKE 사용 (대소문자 구분 없음)
            if q and q.strip():
                search_term = f"%{q.strip()}%"
                # ILIKE는 PostgreSQL에서 대소문자를 구분하지 않는 LIKE
                where_conditions.append("(bill_name ILIKE :search_term OR proposer_name ILIKE :search_term)")
                params["search_term"] = search_term
                print(f"[DEBUG] Search term added: {search_term}", file=sys.stderr, flush=True)
                print(f"[DEBUG] Search term type: {type(search_term)}, encoding: {search_term.encode('utf-8')}", file=sys.stderr, flush=True)
            else:
                print(f"[DEBUG] No search term provided or empty", file=sys.stderr, flush=True)
            
            # 발의일 필터 조건
            if year is not None:
                where_conditions.append("EXTRACT(YEAR FROM propose_dt) = :year")
                params["year"] = int(year)
                print(f"[DEBUG] Year filter: {year}", file=sys.stderr, flush=True)
            if month is not None:
                where_conditions.append("EXTRACT(MONTH FROM propose_dt) = :month")
                params["month"] = int(month)
                print(f"[DEBUG] Month filter: {month}", file=sys.stderr, flush=True)
            if day is not None:
                where_conditions.append("EXTRACT(DAY FROM propose_dt) = :day")
                params["day"] = int(day)
                print(f"[DEBUG] Day filter: {day}", file=sys.stderr, flush=True)
            
            # WHERE 절 구성
            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)
            
            print(f"[DEBUG] WHERE clause: {where_clause}", file=sys.stderr, flush=True)
            print(f"[DEBUG] Params: {params}", file=sys.stderr, flush=True)
            
            # 검색 쿼리 실행 - 더 안전한 방식으로 수정
            base_query = """
                SELECT bill_id, bill_name, proposer_name, propose_dt
                FROM public.final_training_data_copy_sample10_md
            """
            
            if where_clause:
                query = text(base_query + " " + where_clause + " ORDER BY propose_dt DESC, bill_id DESC")
            else:
                query = text(base_query + " ORDER BY propose_dt DESC, bill_id DESC")
            
            results = []
            try:
                print(f"[DEBUG] Executing query...", file=sys.stderr, flush=True)
                print(f"[DEBUG] Query text: {str(query)}", file=sys.stderr, flush=True)
                print(f"[DEBUG] Query params: {params}", file=sys.stderr, flush=True)
                
                df_results = pd.read_sql(query, conn, params=params)
                print(f"[DEBUG] DataFrame shape: {df_results.shape}", file=sys.stderr, flush=True)
                print(f"[DEBUG] DataFrame columns: {df_results.columns.tolist()}", file=sys.stderr, flush=True)
                
                results = df_results.to_dict(orient="records")
                print(f"[DEBUG] Search results count: {len(results)}", file=sys.stderr, flush=True)
                if len(results) > 0:
                    print(f"[DEBUG] First result keys: {list(results[0].keys())}", file=sys.stderr, flush=True)
                    print(f"[DEBUG] First result sample: bill_id={results[0].get('bill_id')}, bill_name={results[0].get('bill_name')[:50] if results[0].get('bill_name') else None}", file=sys.stderr, flush=True)
                else:
                    print(f"[DEBUG] No results found - checking if query is correct...", file=sys.stderr, flush=True)
                    # 간단한 테스트 쿼리 실행
                    test_query = text("SELECT COUNT(*) as cnt FROM public.final_training_data_copy_sample10_md")
                    test_result = pd.read_sql(test_query, conn)
                    total_rows = test_result.iloc[0]['cnt']
                    print(f"[DEBUG] Total rows in table: {total_rows}", file=sys.stderr, flush=True)
                    
                    # 검색어로 직접 테스트
                    if q and q.strip():
                        test_search_query = text("""
                            SELECT COUNT(*) as cnt 
                            FROM public.final_training_data_copy_sample10_md
                            WHERE bill_name ILIKE :search_term OR proposer_name ILIKE :search_term
                        """)
                        test_search_result = pd.read_sql(test_search_query, conn, params={'search_term': f'%{q.strip()}%'})
                        search_count = test_search_result.iloc[0]['cnt']
                        print(f"[DEBUG] Direct search test for '{q}': {search_count} matches found", file=sys.stderr, flush=True)
                        
                        if search_count > 0:
                            # 실제 매칭되는 데이터 샘플 확인
                            sample_query = text("""
                                SELECT bill_id, bill_name, proposer_name 
                                FROM public.final_training_data_copy_sample10_md
                                WHERE bill_name ILIKE :search_term OR proposer_name ILIKE :search_term
                                LIMIT 3
                            """)
                            sample_result = pd.read_sql(sample_query, conn, params={'search_term': f'%{q.strip()}%'})
                            print(f"[DEBUG] Sample matching records:", file=sys.stderr, flush=True)
                            for idx, row in sample_result.iterrows():
                                print(f"  - bill_id: {row['bill_id']}, bill_name: {row['bill_name'][:50]}, proposer: {row['proposer_name']}", file=sys.stderr, flush=True)
            except Exception as db_error:
                print(f"[ERROR] Database query error: {db_error}", file=sys.stderr, flush=True)
                import traceback
                traceback.print_exc(file=sys.stderr)
                results = []
            
            # 날짜 포맷팅 및 JSON 직렬화 가능하도록 변환
            print(f"[DEBUG] Formatting dates for {len(results)} results...", file=sys.stderr, flush=True)
            for b in results:
                pd_val = b.get("propose_dt")
                if pd_val is not None:
                    try:
                        # Timestamp를 문자열로 변환
                        if hasattr(pd_val, "strftime"):
                            date_str = pd_val.strftime("%Y-%m-%d")
                        elif hasattr(pd_val, "isoformat"):
                            date_str = pd_val.isoformat()[:10]
                        else:
                            str_val = str(pd_val)
                            if len(str_val) >= 10:
                                date_str = str_val[:10]
                            else:
                                date_str = ""
                        
                        b["propose_dt_str"] = date_str
                        # 원본 Timestamp 객체를 문자열로 교체 (JSON 직렬화를 위해)
                        b["propose_dt"] = date_str
                    except Exception as e:
                        print(f"Error formatting propose_dt: {e}", file=sys.stderr, flush=True)
                        b["propose_dt_str"] = ""
                        b["propose_dt"] = ""
                else:
                    b["propose_dt_str"] = ""
                    b["propose_dt"] = ""
            
            response_data = {"results": results, "count": len(results)}
            print(f"[DEBUG] Returning response: count={len(results)}", file=sys.stderr, flush=True)
            print(f"[DEBUG] Response data sample: {response_data if len(results) == 0 else {'count': len(results), 'first_result_keys': list(results[0].keys()) if results else []}}", file=sys.stderr, flush=True)
            return JSONResponse(response_data)
    
    except Exception as e:
        print(f"[ERROR] Search error: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return JSONResponse({"results": [], "count": 0})


@app.get("/")

async def main_page(request: Request, page: int = 1):

    limit = 5

    offset = (page - 1) * limit

   

    with engine.connect() as conn:

        # 전체 개수 파악

        total_count = conn.execute(text("SELECT COUNT(*) FROM public.final_training_data_copy_sample10_md")).scalar()

        total_pages = (total_count // limit) + (1 if total_count % limit > 0 else 0)

        # 발의일 필터 옵션: 연 2008~2026, 월 1~12, 일 1~31
        available_years = list(range(2026, 2007, -1))  # 2026, 2025, ..., 2008
        available_months = list(range(1, 13))          # 1~12
        available_days = list(range(1, 32))            # 1~31

        # 목록 조회 (SQL 인젝션 방지를 위해 text() 사용)

        query = text("""

            SELECT bill_id, bill_name, proposer_name, propose_dt

            FROM public.final_training_data_copy_sample10_md

            ORDER BY propose_dt DESC, bill_id DESC

            LIMIT :limit OFFSET :offset

        """)

        latest_bills = pd.read_sql(query, conn, params={"limit": limit, "offset": offset}).to_dict(orient="records")

        for b in latest_bills:
            pd_val = b.get("propose_dt")
            if pd_val is not None:
                try:
                    if hasattr(pd_val, "strftime"):
                        b["propose_dt_str"] = pd_val.strftime("%Y-%m-%d")
                    elif hasattr(pd_val, "isoformat"):
                        b["propose_dt_str"] = pd_val.isoformat()[:10]
                    else:
                        str_val = str(pd_val)
                        if len(str_val) >= 10:
                            b["propose_dt_str"] = str_val[:10]
                        else:
                            b["propose_dt_str"] = ""
                except Exception as e:
                    print(f"Error formatting propose_dt: {e}, value: {pd_val}, type: {type(pd_val)}")
                    b["propose_dt_str"] = ""
            else:
                b["propose_dt_str"] = ""

   

    return templates.TemplateResponse("index.html", {

        "request": request,

        "latest_bills": latest_bills,

        "current_page": page,

        "total_pages": total_pages,

        "available_years": available_years,

        "available_months": available_months,

        "available_days": available_days,

    })



@app.get("/bill/{bill_id}")

async def bill_detail(request: Request, bill_id: str):

    query = text("""

        SELECT 
            f.bill_name, 
            f.proposer_name, 
            f.ai_prediction, 
            f.ai_probability, 
            f.ai_report, 
            f.report_md, 
            f.news,
            c.bill_url

        FROM public.final_training_data_copy_sample10_md f

        LEFT JOIN public.copy_1 c ON f.bill_id = c.bill_id

        WHERE f.bill_id = :bill_id

    """)

   

    # 뉴스 데이터 초기화 (예외 처리 전에 미리 정의)
    news_data = []
    
    try:

        with engine.connect() as conn:

            bill_df = pd.read_sql(query, conn, params={"bill_id": bill_id})

           

        if bill_df.empty:

            return HTMLResponse("<h1>DB에 해당 ID가 없습니다.</h1>", status_code=404)

       

        data = bill_df.iloc[0]

        # 디버깅: bill_url 확인
        bill_url_value = data.get('bill_url', None)
        print(f"[DEBUG] bill_id: {bill_id}")
        print(f"[DEBUG] bill_url 원본 값: {bill_url_value}")
        print(f"[DEBUG] bill_url 타입: {type(bill_url_value)}")
        
        # NaN 체크 및 처리
        if bill_url_value is not None:
            if pd.isna(bill_url_value):
                print(f"[DEBUG] bill_url이 NaN입니다.")
                bill_url_value = None
            else:
                bill_url_value = str(bill_url_value).strip()
                if not bill_url_value:
                    bill_url_value = None
                else:
                    print(f"[DEBUG] bill_url 처리 후 값: {bill_url_value}")
        
        if bill_url_value is None:
            print(f"[WARNING] bill_url이 없습니다. bill_id를 사용하여 기본 URL을 생성합니다.")
            # bill_id를 사용하여 기본 URL 생성
            bill_url_value = f"https://likms.assembly.go.kr/bill/billDetail.do?billId={bill_id}"
            print(f"[DEBUG] 생성된 bill_url: {bill_url_value}")

        # 1. 보고서 파싱 (섹션별로 쪼개기) - report_md 컬럼에서 가져오기

        raw_report_md = data['report_md'] if data['report_md'] else "전문가 분석 보고서가 준비 중입니다."
        
        # 디버깅: report_md 내용 확인
        print(f"[DEBUG] bill_id: {bill_id}")
        print(f"[DEBUG] report_md 타입: {type(raw_report_md)}")
        print(f"[DEBUG] report_md 길이: {len(str(raw_report_md)) if raw_report_md else 0}")
        if raw_report_md:
            # 섹션 5와 6이 있는지 확인
            has_section5 = "## 5." in str(raw_report_md) or "##5." in str(raw_report_md)
            has_section6 = "## 6." in str(raw_report_md) or "##6." in str(raw_report_md)
            print(f"[DEBUG] 섹션 5 포함 여부: {has_section5}")
            print(f"[DEBUG] 섹션 6 포함 여부: {has_section6}")
            if has_section5:
                # 섹션 5 내용 일부 확인
                section5_match = re.search(r'##\s*5\.\s*[^\n]+\n(.*?)(?=##\s*6\.|$)', str(raw_report_md), re.DOTALL)
                if section5_match:
                    section5_content = section5_match.group(1).strip()[:200]
                    print(f"[DEBUG] 섹션 5 내용 일부: {section5_content}")

        parsed_sections = split_expert_report(raw_report_md)

        # 법안 요약에서 "신뢰도" 줄 완전 제거 (후처리)
        _sum = parsed_sections.get("summary") or ""
        if _sum and "신뢰도" in _sum:
            _sum = re.sub(
                r"<li[^>]*>[\s\S]*?신뢰도[\s\S]*?</li>",
                "",
                _sum,
                flags=re.IGNORECASE,
            )
            _sum = re.sub(
                r"<p[^>]*>[\s\S]*?신뢰도[\s\S]*?</p>",
                "",
                _sum,
                flags=re.IGNORECASE,
            )
            parsed_sections["summary"] = _sum

        # 1-1. 예측 확률 테이블 파싱 (도넛 그래프용)

        prediction_data = parse_prediction_table(raw_report_md)

        social_impact_score = parse_social_impact_score(raw_report_md)
        
        # 1-2. 뉴스 데이터 파싱 (유사 사례 비교 분석 섹션용)
        news_data = []
        news_json = data.get('news')
        if news_json:
            try:
                news_list = json.loads(news_json) if isinstance(news_json, str) else news_json
                if isinstance(news_list, list) and len(news_list) > 0:
                    # 최대 10개까지만 사용
                    news_data = news_list[:10]
            except (json.JSONDecodeError, TypeError) as e:
                print(f"[DEBUG] 뉴스 데이터 파싱 오류: {e}")
                news_data = []



        # 2. ai_report를 HTML로 변환 및 파싱

        ai_report_html = ""
        proposal_reason_html = ""
        main_content_html = ""

        ai_report_raw = data.get('ai_report')

        print(f"[DEBUG] ai_report_raw 타입: {type(ai_report_raw)}, 값: {str(ai_report_raw)[:100] if ai_report_raw else 'None'}")

       

        if ai_report_raw is not None and str(ai_report_raw).strip():

            ai_report_text = str(ai_report_raw).strip()

            print(f"[DEBUG] ai_report 길이: {len(ai_report_text)}")

            try:

                md = markdown.Markdown(extensions=['extra', 'nl2br', 'sane_lists', 'tables'])

                ai_report_html = md.convert(ai_report_text)
                
                # ai_report에서 "제안 이유"와 "주요 내용" 파싱
                # 더 간단하고 확실한 방법: 줄바꿈으로 분리하고 각 섹션 찾기
                print(f"[DEBUG] ai_report_text 전체 (처음 500자): {repr(ai_report_text[:500])}")
                
                # 정규화: 모든 줄바꿈을 \n으로 통일
                normalized_text = ai_report_text.replace('\r\n', '\n').replace('\r', '\n')
                
                # 제안 이유 파싱
                proposal_reason_match = re.search(
                    r'제안\s*이유\s*:?\s*(.+?)(?=\n\s*주요\s*내용:|주요\s*내용\s*:|시사점\s*:|$)',
                    normalized_text,
                    re.DOTALL | re.IGNORECASE
                )
                
                # 주요 내용 파싱
                main_content_match = re.search(
                    r'주요\s*내용\s*:?\s*(.+?)(?=\n\s*시사점\s*:|시사점\s*:|$)',
                    normalized_text,
                    re.DOTALL | re.IGNORECASE
                )
                
                print(f"[DEBUG] 제안 이유 매칭: {proposal_reason_match is not None}")
                print(f"[DEBUG] 주요 내용 매칭: {main_content_match is not None}")
                
                if proposal_reason_match:
                    proposal_reason_text = proposal_reason_match.group(1).strip()
                else:
                    # Fallback: 줄바꿈으로 분리해서 직접 찾기
                    lines = normalized_text.split('\n')
                    proposal_start_idx = None
                    main_start_idx = None
                    for i, line in enumerate(lines):
                        if re.search(r'제안\s*이유\s*:', line, re.IGNORECASE):
                            proposal_start_idx = i
                        if re.search(r'주요\s*내용\s*:', line, re.IGNORECASE):
                            main_start_idx = i
                            break
                    
                    if proposal_start_idx is not None and main_start_idx is not None:
                        # 제안 이유: proposal_start_idx 다음 줄부터 main_start_idx 전까지
                        proposal_lines = lines[proposal_start_idx + 1:main_start_idx]
                        proposal_reason_text = '\n'.join(proposal_lines).strip()
                        # 첫 줄에서 "제안 이유:" 제거
                        first_line = lines[proposal_start_idx]
                        proposal_reason_text = re.sub(r'제안\s*이유\s*:?\s*', '', first_line, flags=re.IGNORECASE).strip() + '\n' + proposal_reason_text
                        proposal_reason_text = proposal_reason_text.strip()
                        print(f"[DEBUG] 제안 이유 Fallback 파싱 성공, 길이: {len(proposal_reason_text)}")
                    else:
                        proposal_reason_text = ""
                        print(f"[DEBUG] 제안 이유 Fallback 파싱 실패")
                
                if proposal_reason_text:
                    print(f"[DEBUG] 제안 이유 텍스트 길이: {len(proposal_reason_text)}, 내용: {proposal_reason_text[:100]}")
                    proposal_reason_html = md.convert(proposal_reason_text)
                    # HTML 태그 정리
                    proposal_reason_html = re.sub(r'</?strong[^>]*>', '', proposal_reason_html, flags=re.IGNORECASE)
                    proposal_reason_html = re.sub(r'</?b[^>]*>', '', proposal_reason_html, flags=re.IGNORECASE)
                    proposal_reason_html = proposal_reason_html.strip()
                    print(f"[DEBUG] 제안 이유 HTML 생성 완료, 길이: {len(proposal_reason_html)}")
                else:
                    print(f"[DEBUG] 제안 이유 텍스트가 비어있음")
                
                if main_content_match:
                    main_content_text = main_content_match.group(1).strip()
                else:
                    # Fallback: 줄바꿈으로 분리해서 직접 찾기
                    lines = normalized_text.split('\n')
                    main_start_idx = None
                    for i, line in enumerate(lines):
                        if re.search(r'주요\s*내용\s*:', line, re.IGNORECASE):
                            main_start_idx = i
                            break
                    
                    if main_start_idx is not None:
                        # 주요 내용: main_start_idx 다음 줄부터 끝까지
                        main_lines = lines[main_start_idx + 1:]
                        main_content_text = '\n'.join(main_lines).strip()
                        # 첫 줄에서 "주요 내용:" 제거
                        first_line = lines[main_start_idx]
                        main_content_text = re.sub(r'주요\s*내용\s*:?\s*', '', first_line, flags=re.IGNORECASE).strip() + '\n' + main_content_text
                        main_content_text = main_content_text.strip()
                        print(f"[DEBUG] 주요 내용 Fallback 파싱 성공, 길이: {len(main_content_text)}")
                    else:
                        main_content_text = ""
                        print(f"[DEBUG] 주요 내용 Fallback 파싱 실패")
                
                if main_content_text:
                    print(f"[DEBUG] 주요 내용 텍스트 길이: {len(main_content_text)}, 내용: {main_content_text[:100]}")
                    main_content_html = md.convert(main_content_text)
                    # HTML 태그 정리
                    main_content_html = re.sub(r'</?strong[^>]*>', '', main_content_html, flags=re.IGNORECASE)
                    main_content_html = re.sub(r'</?b[^>]*>', '', main_content_html, flags=re.IGNORECASE)
                    main_content_html = main_content_html.strip()
                    print(f"[DEBUG] 주요 내용 HTML 생성 완료, 길이: {len(main_content_html)}")
                else:
                    print(f"[DEBUG] 주요 내용 텍스트가 비어있음")

               

                # 볼드 태그 제거 (strong, b 태그를 일반 텍스트로)
                # <strong>태그와 <b>태그 제거 (내용은 유지)

                ai_report_html = re.sub(r'</?strong[^>]*>', '', ai_report_html, flags=re.IGNORECASE)

                ai_report_html = re.sub(r'</?b[^>]*>', '', ai_report_html, flags=re.IGNORECASE)

               

                # 1. 맨 윗줄에 법안명 볼드체로 추가 (인라인 style로 확실히 적용) - 제거됨
                # bill_name = str(data['bill_name'])
                # from markupsafe import escape
                # bill_name_safe = escape(bill_name)
                # ai_report_html = f'<p><strong class="bill-name-bold" style="font-weight: 700 !important;">{bill_name_safe}</strong></p>\n' + ai_report_html

               

                # 2. "제안 이유:", "주요 내용:", "시사점:" 같은 섹션 제목을 볼드체로 강조
                # 역순으로 처리하여 나중 섹션이 먼저 섹션에 포함되지 않도록 함

                section_titles = ['시사점:', '주요 내용:', '제안 이유:']  # 역순

                for title in section_titles:

                    # 이미 strong.section-title-bold로 감싸져 있으면 스킵

                    if f'<strong class="section-title-bold">{title}</strong>' in ai_report_html:

                        print(f"[DEBUG] '{title}' 이미 처리됨 - 스킵")

                        continue

                   

                    # "주요 내용:"과 "시사점:" 앞에 개행 추가할지 결정

                    needs_break_before = title in ['주요 내용:', '시사점:']

                    break_before = '<br><br>' if needs_break_before else ''

                   

                    # 가장 간단한 패턴: "제안 이유:" 다음에 공백이 있거나 없거나, 다른 섹션 제목이나 태그 끝까지

                    # 공백 있는 경우: "제안 이유: 내용"

                    pattern1 = re.escape(title) + r'(\s+)([^<]+?)(?=주요\s*내용:|시사점:|제안\s*이유:|<|$)'

                    def replace1(m):

                        space = m.group(1)

                        content = m.group(2)

                        return f'{break_before}<strong class="section-title-bold">{title}</strong>{space}{content}'

                    new_html = re.sub(pattern1, replace1, ai_report_html, flags=re.IGNORECASE)

                    if new_html != ai_report_html:

                        ai_report_html = new_html

                        print(f"[DEBUG] '{title}' 볼드 처리 완료 (공백 있음)")

                        continue

                   

                    # 공백 없는 경우: "제안 이유:내용"

                    pattern2 = re.escape(title) + r'([^<\s][^<]+?)(?=주요\s*내용:|시사점:|제안\s*이유:|<|$)'

                    def replace2(m):

                        content = m.group(1)

                        return f'{break_before}<strong class="section-title-bold">{title}</strong> {content}'

                    new_html = re.sub(pattern2, replace2, ai_report_html, flags=re.IGNORECASE)

                    if new_html != ai_report_html:

                        ai_report_html = new_html

                        print(f"[DEBUG] '{title}' 볼드 처리 완료 (공백 없음)")

                    else:

                        print(f"[DEBUG] '{title}' 볼드 처리 실패 - 패턴 매칭 실패")

               

                # 3. 추가 개행 처리: 이미 볼드 처리된 섹션 사이에 개행이 없으면 추가

                # "제안 이유:" 다음에 "주요 내용:"이 같은 <p> 안에 있는 경우

                ai_report_html = re.sub(r'(<strong[^>]*>제안\s*이유:</strong>[^<]+?)(<strong[^>]*>주요\s*내용:</strong>)', r'\1<br><br>\2', ai_report_html, flags=re.IGNORECASE)

                # "주요 내용:" 다음에 "시사점:"이 같은 <p> 안에 있는 경우

                ai_report_html = re.sub(r'(<strong[^>]*>주요\s*내용:</strong>[^<]+?)(<strong[^>]*>시사점:</strong>)', r'\1<br><br>\2', ai_report_html, flags=re.IGNORECASE)

               

                # 중복된 <br><br> 제거 (3개 이상 연속된 경우 2개로)

                ai_report_html = re.sub(r'(<br>\s*){3,}', '<br><br>', ai_report_html, flags=re.IGNORECASE)

               

                # "핵심 요약 리포트:" 부분만 강조 처리
                # 가장 확실한 방법: 첫 번째 <p> 태그에서 "핵심 요약 리포트:" 찾아서 치환

                # 패턴 1: <p>태그 안에 있는 경우 (가장 일반적)

                pattern1 = r'(<p[^>]*>)(핵심\s*요약\s*리포트\s*:)\s*([^<]+?)(</p>)'

                match1 = re.search(pattern1, ai_report_html, flags=re.IGNORECASE)

               

                if match1:

                    p_open = match1.group(1)

                    title = match1.group(2)

                    rest = match1.group(3)

                    p_close = match1.group(4)

                    replacement = f'{p_open}<span class="report-title-highlight">{title}</span>{rest}{p_close}'

                    ai_report_html = ai_report_html.replace(match1.group(0), replacement, 1)

                    print(f"[DEBUG] '핵심 요약 리포트:' 강조 처리 완료 (패턴1)")

                else:

                    # 패턴 2: 첫 번째 <p> 태그 전체를 찾아서 치환

                    first_p_pattern = r'<p[^>]*>.*?핵심\s*요약\s*리포트\s*:.*?</p>'

                    match2 = re.search(first_p_pattern, ai_report_html, flags=re.IGNORECASE | re.DOTALL)

                   

                    if match2:

                        original_p = match2.group(0)

                        # "핵심 요약 리포트:" 부분만 span으로 감싸기

                        replaced_p = original_p.replace('핵심 요약 리포트:', '<span class="report-title-highlight">핵심 요약 리포트:</span>', 1)

                        ai_report_html = ai_report_html.replace(original_p, replaced_p, 1)

                        print(f"[DEBUG] '핵심 요약 리포트:' 강조 처리 완료 (패턴2)")

                    else:

                        # 패턴 3: 단순 문자열 치환 (최후의 수단)

                        if '핵심 요약 리포트:' in ai_report_html:

                            ai_report_html = ai_report_html.replace('핵심 요약 리포트:', '<span class="report-title-highlight">핵심 요약 리포트:</span>', 1)

                            print(f"[DEBUG] '핵심 요약 리포트:' 강조 처리 완료 (패턴3 - 단순 치환)")

                        else:

                            print(f"[DEBUG] '핵심 요약 리포트:' 강조 처리 실패")

                            print(f"[DEBUG] HTML 일부 (처음 500자): {ai_report_html[:500]}")

                            print(f"[DEBUG] '핵심' 포함 여부: {'핵심' in ai_report_html}")

                            print(f"[DEBUG] '요약' 포함 여부: {'요약' in ai_report_html}")

                            if '핵심' in ai_report_html:

                                # 부분 매칭으로 찾기

                                핵심_idx = ai_report_html.find('핵심')

                                print(f"[DEBUG] '핵심' 위치 주변 텍스트: {ai_report_html[max(0, 핵심_idx-50):핵심_idx+100]}")

               

                print(f"[DEBUG] ai_report_html 길이: {len(ai_report_html)}")

            except Exception as e:

                print(f"[ERROR] ai_report 마크다운 변환 오류: {str(e)}")

                ai_report_html = f"<p>{ai_report_text}</p>"  # 마크다운 변환 실패 시 일반 텍스트로

        else:

            print(f"[DEBUG] ai_report가 비어있거나 None입니다.")

       

        # 3. 결과물 조합

        context = {

            "request": request,

            "bill_id": bill_id,

            "bill_name": str(data['bill_name']),

            "proposer_name": str(data['proposer_name']),

            "probability": str(data['ai_probability']) if data['ai_probability'] is not None else "0.0",

            "prediction": str(data['ai_prediction']) if data['ai_prediction'] else "데이터 없음",

            "report": str(data['ai_report']) if data['ai_report'] else "",

            "ai_report_html": ai_report_html,  # HTML로 변환된 ai_report
            "proposal_reason_html": proposal_reason_html,  # 제안 이유 HTML
            "main_content_html": main_content_html,  # 주요 내용 HTML

            "sections": parsed_sections,  # 👈 쪼개진 HTML 섹션들이 담긴 딕셔너리
            "news_data": news_data,  # 👈 뉴스 데이터 리스트 (최대 10개)

            "prediction_data": json.dumps(prediction_data, ensure_ascii=False),  # 👈 도넛 그래프용 예측 확률 데이터 (JSON 문자열)
            "social_impact_score": social_impact_score,  # 👈 사회적 영향력 점수 (0~100, 반원 게이지용)
            "bill_url": bill_url_value,  # 👈 원문 링크 URL (항상 올바른 값으로 설정됨)

        }
        
        # 디버깅: context에 전달되는 bill_url 확인
        print(f"[DEBUG] context에 전달되는 bill_url: {context.get('bill_url', 'NOT_FOUND')}")

       

        # 디버깅: 파싱 결과 확인

        print(f"[DEBUG] bill_id: {bill_id}")

        print(f"[DEBUG] report_md 길이: {len(raw_report_md) if raw_report_md else 0}")

        print(f"[DEBUG] sections keys: {list(parsed_sections.keys())}")

        for k in ['summary', 'bill_info', 'prediction', 'evidence', 'similar_cases', 'social_impact', 'next_action']:

            v = parsed_sections.get(k, '')

            has_content = bool(v and v.strip())

            print(f"[DEBUG] {k}: length={len(v)}, has_content={has_content}")

            if not has_content:

                print(f"[DEBUG] ⚠️ {k} is EMPTY!")



        return templates.TemplateResponse("detail.html", context)



    except Exception as e:

        print(f"❌ 에러 발생: {str(e)}")

        return HTMLResponse(content=f"<h1>서버 오류</h1><p>{str(e)}</p>", status_code=500)


def search_web(query: str, max_results: int = 5) -> list:
    print(f"[DEBUG] 실제 검색 쿼리: {query}")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            print(f"[DEBUG] 검색 결과 개수: {len(results)}")
            return results
    except Exception as e:
        print(f"웹 검색 오류 발생: {e}")
        return []


# ==================== RAG (Retrieval-Augmented Generation) 구현 ====================

def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """
    텍스트를 벡터 임베딩으로 변환
    """
    try:
        response = openai.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"⚠️ 임베딩 생성 실패: {e}")
        return None


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    긴 텍스트를 작은 청크로 분할 (오버랩 포함)
    """
    if not text or len(text) < chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # 문장 경계에서 자르기 (마지막 문장이 완전하지 않으면)
        if end < len(text):
            # 마지막 마침표, 줄바꿈, 또는 공백에서 자르기
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            last_space = chunk.rfind(' ')
            
            cut_point = max(last_period, last_newline, last_space)
            if cut_point > chunk_size * 0.7:  # 너무 앞에서 자르지 않도록
                chunk = chunk[:cut_point + 1]
                end = start + cut_point + 1
        
        chunks.append(chunk.strip())
        start = end - overlap  # 오버랩으로 다음 청크 시작
    
    return chunks


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    코사인 유사도 계산
    """
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def retrieve_relevant_context(
    question: str,
    bill_name: str,
    engine,
    top_k: int = 3
) -> List[Dict[str, any]]:
    """
    RAG: 질문과 관련된 법안 문서를 벡터 검색으로 찾기
    
    Returns:
        List[Dict]: [{"text": "...", "source": "...", "score": 0.xx}, ...]
    """
    try:
        print(f"🔍 RAG 검색 시작: {question}")
        
        # 1. 질문을 임베딩으로 변환
        question_embedding = get_embedding(question)
        if not question_embedding:
            print("⚠️ 질문 임베딩 생성 실패")
            return []
        
        # 2. DB에서 법안 관련 문서 조회
        with engine.connect() as conn:
            query = text("""
                SELECT bill_name, summary, report_md, ai_report, news
                FROM public.final_training_data_copy_sample10_md 
                WHERE bill_name = :bill_name 
                LIMIT 1
            """)
            bill_df = pd.read_sql(query, conn, params={"bill_name": bill_name})
            
            if bill_df.empty:
                print("⚠️ 법안 정보를 찾을 수 없습니다")
                return []
            
            row = bill_df.iloc[0]
            
            # 3. 문서들을 청크로 분할하고 임베딩 생성
            documents = []
            
            # summary 청크
            if row.get('summary'):
                summary_chunks = chunk_text(str(row['summary']))
                for i, chunk in enumerate(summary_chunks):
                    if chunk:
                        documents.append({
                            'text': chunk,
                            'source': f'summary_chunk_{i}',
                            'type': 'summary'
                        })
            
            # report_md 청크 (큰 문서이므로 여러 청크로)
            if row.get('report_md'):
                report_text = str(row['report_md'])
                # 마크다운 제목 제거 (더 깔끔한 텍스트)
                report_text = re.sub(r'#+\s+', '', report_text)
                report_text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', report_text)
                report_chunks = chunk_text(report_text, chunk_size=600, overlap=100)
                for i, chunk in enumerate(report_chunks):
                    if chunk and len(chunk) > 100:  # 너무 짧은 청크 제외
                        documents.append({
                            'text': chunk,
                            'source': f'report_md_chunk_{i}',
                            'type': 'report'
                        })
            
            # ai_report 청크
            if row.get('ai_report'):
                ai_report_text = str(row['ai_report'])
                ai_report_chunks = chunk_text(ai_report_text, chunk_size=400)
                for i, chunk in enumerate(ai_report_chunks):
                    if chunk:
                        documents.append({
                            'text': chunk,
                            'source': f'ai_report_chunk_{i}',
                            'type': 'ai_report'
                        })
            
            # news 청크 (뉴스가 여러 개일 수 있음)
            if row.get('news'):
                try:
                    news_data = row['news']
                    if isinstance(news_data, str):
                        news_list = json.loads(news_data)
                    else:
                        news_list = news_data
                    
                    if isinstance(news_list, list):
                        for i, news_item in enumerate(news_list[:5]):  # 최대 5개 뉴스
                            news_text = f"{news_item.get('title', '')} {news_item.get('summary', '')} {news_item.get('body', '')}"
                            if news_text.strip():
                                documents.append({
                                    'text': news_text[:500],  # 뉴스는 짧게
                                    'source': f'news_{i}',
                                    'type': 'news'
                                })
                except:
                    pass
            
            if not documents:
                print("⚠️ 검색할 문서가 없습니다")
                return []
            
            print(f"📚 총 {len(documents)}개 문서 청크 생성")
            
            # 4. 각 문서 청크를 임베딩으로 변환하고 유사도 계산
            scored_docs = []
            for doc in documents:
                doc_embedding = get_embedding(doc['text'])
                if doc_embedding:
                    similarity = cosine_similarity(question_embedding, doc_embedding)
                    scored_docs.append({
                        'text': doc['text'],
                        'source': doc['source'],
                        'type': doc['type'],
                        'score': similarity
                    })
            
            # 5. 유사도 순으로 정렬하고 상위 k개 선택
            scored_docs.sort(key=lambda x: x['score'], reverse=True)
            top_docs = scored_docs[:top_k]
            
            print(f"✅ RAG 검색 완료: 상위 {len(top_docs)}개 문서 선택")
            for i, doc in enumerate(top_docs):
                print(f"  {i+1}. [{doc['type']}] 유사도: {doc['score']:.3f} - {doc['text'][:50]}...")
            
            return top_docs
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ RAG 검색 오류: {e}")
        return []


def build_enhanced_prompt(
    question: str,
    question_intent: str,
    bill_name: str,
    display_prob: float,
    pred_value: str,
    proposer_info: str,
    ctx: str,
    search_summary: str = "",
    has_web_search: bool = False,
    rag_context: str = ""
) -> str:
    """
    질문 유형별 맞춤형 고품질 프롬프트 생성
    """
    # 발의자 이름 추출 (예시용)
    proposer_name_for_example = ""
    if proposer_info and "발의자:" in proposer_info:
        try:
            proposer_name_for_example = proposer_info.split("발의자:")[1].split("\n")[0].strip()
        except:
            proposer_name_for_example = "정보 없음"
    
    # 질문 유형별 Few-shot 예시
    examples = {
        'probability': {
            'example_q': '이 법안이 통과될 확률이 얼마나 되나요?',
            'example_a': f'이 법안의 통과 확률은 {display_prob}%로 예측됩니다. 예측 결과는 {pred_value}입니다.'
        },
        'proposer': {
            'example_q': '이 법안의 발의자는 누구인가요?',
            'example_a': f'이 법안은 {proposer_name_for_example if proposer_name_for_example else "정보 없음"}이(가) 발의했습니다.'
        },
        'other_bills': {
            'example_q': '이 발의자가 발의한 다른 법안에는 무엇이 있나요?',
            'example_a': '해당 발의자가 발의한 다른 법안으로는 [DB에서 조회한 법안 목록]이 있습니다.'
        },
        'news': {
            'example_q': '이 법안과 관련된 뉴스가 있나요?',
            'example_a': '이 법안과 관련하여 [DB 뉴스 또는 웹 검색 결과]가 보도되었습니다.'
        },
        'content': {
            'example_q': '이 법안의 주요 내용은 무엇인가요?',
            'example_a': '이 법안은 [참고 내용에서 요약]을 주요 내용으로 합니다.'
        },
        'status': {
            'example_q': '이 법안의 처리 현황은 어떻게 되나요?',
            'example_a': '현재 이 법안은 [웹 검색 결과 또는 DB 정보] 상태입니다.'
        }
    }
    
    # 기본 정보 구성
    base_info = f"""법안명: {bill_name}
예측 확률: {display_prob}%
예측 결과: {pred_value}
{proposer_info}"""
    
    if ctx:
        base_info += f"\n법안 요약: {ctx[:500]}"
    
    # 질문 유형별 맞춤 지침
    intent_guidelines = {
        'probability': """답변 형식:
1. 확률 값과 예측 결과를 명확히 제시 (1-2문장)
2. 간단한 해석 (1문장, 선택적)""",
        
        'proposer': """답변 형식:
1. 발의자 정보 명확히 제시 (1문장)
2. 발의한 다른 법안 언급 (있는 경우, 1문장)""",
        
        'other_bills': """답변 형식:
1. 발의자 이름과 다른 법안 목록 제시 (최대 3개, 1-2문장)""",
        
        'news': """답변 형식:
1. 관련 뉴스 요약 (1-2개, 1-2문장)
2. 주요 이슈 언급 (1문장, 선택적)""",
        
        'content': """답변 형식:
1. 법안의 핵심 목적과 주요 내용 요약 (1-2문장)""",
        
        'status': """답변 형식:
1. 현재 처리 상태를 명확히 제시 (1-2문장)
2. 최신 정보 언급 (있는 경우, 1문장)"""
    }
    
    guideline = intent_guidelines.get(question_intent, "질문에 정확하고 도움이 되는 답변을 제공하세요.")
    
    # Few-shot 예시 선택
    example = examples.get(question_intent, {})
    example_text = ""
    if example:
        example_text = f"""
[좋은 답변 예시]
질문: {example.get('example_q', '')}
답변: {example.get('example_a', '')}
"""
    
    # 웹 검색 결과 섹션
    web_search_section = ""
    if has_web_search and search_summary:
        web_search_section = f"""
[웹 검색 결과 (최신 정보 보완용)]:
{search_summary}

중요: 웹 검색 결과는 DB 정보를 보완하는 용도입니다. DB 정보가 있으면 우선 사용하세요.
"""
    
    # RAG 검색 결과 섹션 (벡터 검색으로 찾은 관련 문서)
    rag_section = ""
    if rag_context:
        rag_section = f"""
[RAG 검색 결과 (질문과 가장 관련성 높은 법안 문서)]:
{rag_context}

중요: RAG 검색 결과는 질문과의 유사도가 높은 법안 문서입니다. 이 정보를 우선적으로 활용하여 정확한 답변을 제공하세요.
"""
    
    # 최종 프롬프트 구성
    prompt = f"""당신은 법률 전문가 'LawDict AI'입니다. 법안 '{bill_name}'에 대한 전문가 상담을 제공합니다.

[법안 정보]
{base_info}
{rag_section}
{web_search_section}
[답변 지침]
{guideline}
{example_text}
[응답 규칙]
1. 정확성: 제공된 정보만 사용하고 추측하지 마세요
2. 명확성: 구체적인 수치와 사실을 명확히 제시하세요
3. 간결성: 1-3문장으로 핵심만 전달하세요
4. 자연스러움: 자연스러운 한국어로 답변하세요
5. 형식: 순수 텍스트만 사용 (마크다운, 특수문자 제거)
6. 정보 우선순위: RAG 검색 결과 > DB 정보 > 웹 검색 결과 > 일반 지식

현재 질문: "{question}"
위 지침을 따라 정확하고 도움이 되는 답변을 제공하세요."""
    
    return prompt


def validate_and_improve_response(
    response: str,
    question_intent: str,
    display_prob: float,
    pred_value: str,
    proposer_name: str = ""
) -> str:
    """
    응답 품질 검증 및 개선
    """
    if not response or len(response.strip()) < 10:
        return "죄송합니다. 답변을 생성하는 데 문제가 발생했습니다. 다시 질문해 주세요."
    
    # 응답 길이 검증 (너무 짧거나 길면 조정)
    if len(response) < 20:
        # 너무 짧은 경우 보완
        if question_intent == 'probability':
            response = f"이 법안의 통과 확률은 {display_prob}%로 예측되며, 예측 결과는 {pred_value}입니다. {response}"
        elif question_intent == 'proposer' and proposer_name:
            response = f"이 법안의 발의자는 {proposer_name}입니다. {response}"
    
    if len(response) > 400:
        # 너무 긴 경우 요약 (2문장으로 제한)
        sentences = response.split('。')
        if not sentences or len(sentences) == 1:
            sentences = response.split('.')
        if not sentences or len(sentences) == 1:
            sentences = response.split('?')
        response = '. '.join(sentences[:2]) + '.'
    
    # 확률 질문인데 확률이 없으면 추가
    if question_intent == 'probability' and str(display_prob) not in response and display_prob > 0:
        response = f"통과 확률은 {display_prob}%입니다. {response}"
    
    # 발의자 질문인데 발의자 이름이 없으면 추가
    if question_intent == 'proposer' and proposer_name and proposer_name not in response:
        response = f"발의자는 {proposer_name}입니다. {response}"
    
    return response.strip()


def classify_question_intent(question: str, bill_name: str, proposer_name: str = "") -> dict:
    """
    질문 의도를 분류하는 함수
    키워드 기반 + LLM을 활용하여 질문 유형을 정확하게 분류
    
    Returns:
        dict: {
            'intent': 'probability' | 'proposer' | 'other_bills' | 'news' | 'status' | 'content' | 'general',
            'confidence': float (0-1),
            'needs_db': bool,
            'needs_web_search': bool,
            'priority': int (1-5, 높을수록 우선순위 높음)
        }
    """
    question_lower = question.lower()
    
    # 1단계: 키워드 기반 빠른 분류 (성능 최적화)
    intent_keywords = {
        'probability': ['확률', '가능성', '예측', '결과', '얼마나', '퍼센트', '%', '가능'],
        'proposer': ['발의자', '발의한', '제안한', '누구', '뭘하는', '정당', '소속', '경력', '이력', '의원'],
        'other_bills': ['다른 법안', '다른 제안', '다른 법률안', '다른 법안에는', '발의한 법안'],
        'news': ['뉴스', '논란', '이슈', '반대', '찬성', '의견', '관련 뉴스', '보도', '기사'],
        'status': ['통과', '처리', '현황', '상태', '진행', '최신', '결과', '처리됨', '통과됨'],
        'content': ['내용', '무엇', '어떤', '요약', '설명', '주요', '핵심'],
    }
    
    # 각 의도별 매칭 점수 계산
    intent_scores = {}
    for intent, keywords in intent_keywords.items():
        score = sum(1 for keyword in keywords if keyword in question_lower)
        if score > 0:
            intent_scores[intent] = score
    
    # 최고 점수 의도 선택
    primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0] if intent_scores else 'general'
    
    # 2단계: LLM을 활용한 정교한 분류 (복합 질문 처리)
    try:
        classification_prompt = f"""다음 질문을 분석하여 의도를 분류하세요.

질문: "{question}"
법안명: "{bill_name}"
발의자: "{proposer_name if proposer_name else '정보 없음'}"

의도 유형:
1. probability: 확률/예측 결과 질문
2. proposer: 발의자 정보 질문
3. other_bills: 발의한 다른 법안 질문
4. news: 뉴스/논란/이슈 질문
5. status: 처리 현황/상태 질문
6. content: 법안 내용/요약 질문
7. general: 일반 질문

JSON 형식으로만 답변하세요:
{{"intent": "의도유형", "confidence": 0.0-1.0, "needs_db": true/false, "needs_web_search": true/false, "priority": 1-5}}"""

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 질문 의도 분류 전문가입니다. JSON 형식으로만 답변하세요."},
                {"role": "user", "content": classification_prompt}
            ],
            temperature=0.1,
            max_tokens=150,
            response_format={"type": "json_object"}
        )
        
        llm_result = json.loads(response.choices[0].message.content)
        
        # 키워드 기반 결과와 LLM 결과 결합 (LLM 결과 우선, 키워드 결과로 검증)
        final_intent = llm_result.get('intent', primary_intent)
        
        # 키워드 기반 결과와 LLM 결과가 다르면 신뢰도 낮춤
        if final_intent != primary_intent and primary_intent != 'general':
            confidence = min(llm_result.get('confidence', 0.7), 0.7)
        else:
            confidence = llm_result.get('confidence', 0.9)
        
        result = {
            'intent': final_intent,
            'confidence': confidence,
            'needs_db': llm_result.get('needs_db', True),
            'needs_web_search': llm_result.get('needs_web_search', False),
            'priority': llm_result.get('priority', 3)
        }
        
        print(f"🎯 질문 분류: {result['intent']} (신뢰도: {result['confidence']:.2f}, 우선순위: {result['priority']})")
        return result
        
    except Exception as e:
        print(f"⚠️ LLM 분류 실패, 키워드 기반 사용: {e}")
        # LLM 실패 시 키워드 기반 결과 사용
        needs_db = primary_intent != 'general'
        needs_web_search = primary_intent in ['proposer', 'other_bills', 'news', 'status']
        priority = 5 if primary_intent == 'probability' else (4 if primary_intent in ['proposer', 'other_bills'] else 3)
        
        return {
            'intent': primary_intent,
            'confidence': 0.7,
            'needs_db': needs_db,
            'needs_web_search': needs_web_search,
            'priority': priority
        }


# 3. 챗봇 실행 구역 (이거 하나만 남기세요!)
@app.post("/api/chat")
async def chat_with_lawdict(data: ChatRequest):
    try:
        # 데이터 수신 로그 확인 (터미널에 뜨는지 보세요!)
        print(f"📡 수신: {data.bill_name} | 질문: {data.message}")

        # 보고서(report_md)에서 확률 정보 파싱
        display_prob = 0.0
        pred_value = "데이터 없음"
        
        # DB에서 발의자 정보 및 보고서 조회
        proposer_name = ""
        db_info = {}
        
        try:
            with engine.connect() as conn:
                # 발의자 정보와 report_md 함께 조회
                query = text("""
                    SELECT proposer_name, report_md, propose_dt, summary
                    FROM public.final_training_data_copy_sample10_md 
                    WHERE bill_name = :bill_name 
                    LIMIT 1
                """)
                bill_df = pd.read_sql(query, conn, params={"bill_name": data.bill_name})
                
                if not bill_df.empty:
                    row = bill_df.iloc[0]
                    proposer_name = str(row.get('proposer_name', '') or '')
                    report_md = str(row.get('report_md', ''))
                    db_info = {
                        'proposer_name': proposer_name,
                        'propose_dt': str(row.get('propose_dt', '')),
                        'summary': str(row.get('summary', ''))
                    }
                    
                    # report_md에서 "모델 예측 결과" 테이블 파싱
                    # 테이블 형식: | 순위 | 예측 | 확률 |
                    # 예: | 1 | 대안반영폐기 | 0.3995 |
                    import re
                    
                    # "모델 예측 결과" 섹션 찾기
                    prediction_section = re.search(
                        r'##\s*2\.\s*모델\s*예측\s*결과.*?(?=##|$)',
                        report_md,
                        re.DOTALL | re.IGNORECASE
                    )
                    
                    if prediction_section:
                        section_text = prediction_section.group(0)
                        
                        # 테이블에서 첫 번째 행(가장 높은 확률) 찾기
                        # | 1 | 예측명 | 확률값 |
                        table_row = re.search(
                            r'\|\s*1\s*\|\s*([^|]+)\s*\|\s*([0-9.]+)\s*\|',
                            section_text,
                            re.IGNORECASE
                        )
                        
                        if table_row:
                            pred_value = table_row.group(1).strip()
                            prob_decimal = float(table_row.group(2).strip())
                            display_prob = round(prob_decimal * 100, 1)
                            print(f"✅ 보고서에서 파싱: {pred_value} {display_prob}%")
                        else:
                            # 테이블 형식이 다를 수 있으니 다른 패턴 시도
                            # 확률 값 찾기 (0.xxxx 형식)
                            prob_match = re.search(r'0\.\d{3,4}', section_text)
                            if prob_match:
                                prob_decimal = float(prob_match.group(0))
                                display_prob = round(prob_decimal * 100, 1)
                                
                            # 예측명 찾기
                            pred_match = re.search(
                                r'(임기만료폐기|원안가결|수정가결|대안반영폐기|수정안반영폐기|부결|폐기|철회)',
                                section_text
                            )
                            if pred_match:
                                pred_value = pred_match.group(1)
                    
                    # report_md에서 찾지 못했으면 context에서 시도
                    if display_prob == 0.0 and data.context:
                        # context에서 확률 퍼센트 찾기 (예: 40.0%)
                        prob_match = re.search(r'(\d+\.?\d*)%', data.context)
                        if prob_match:
                            display_prob = float(prob_match.group(1))
                            
                        # context에서 예측명 찾기
                        if pred_value == "데이터 없음":
                            pred_match = re.search(
                                r'(임기만료폐기|원안가결|수정가결|대안반영폐기|수정안반영폐기|부결|폐기|철회)',
                                data.context
                            )
                            if pred_match:
                                pred_value = pred_match.group(1)
                                
        except Exception as db_err:
            print(f"⚠️ 보고서 파싱 실패: {db_err}")
            # Fallback: context에서 확률 찾기
            if data.context:
                import re
                prob_match = re.search(r'(\d+\.?\d*)%', data.context)
                if prob_match:
                    display_prob = float(prob_match.group(1))

        ctx = (data.context or "")[:1500]
        
        # 질문 의도 분류 (개선된 시스템)
        intent_result = classify_question_intent(data.message, data.bill_name, proposer_name)
        question_intent = intent_result['intent']
        needs_db = intent_result['needs_db']
        needs_web_search = intent_result['needs_web_search']
        intent_priority = intent_result['priority']
        
        # 호환성을 위한 플래그 (기존 코드와의 호환)
        is_proposer_question = question_intent == 'proposer'
        is_other_bills_question = question_intent == 'other_bills'
        is_news_question = question_intent == 'news'
        is_status_question = question_intent == 'status'
        is_probability_question = question_intent == 'probability'
        is_content_question = question_intent == 'content'
        
        # 모든 질문에 대해 DB에서 관련 정보 조회
        other_bills = []
        db_news = []
        db_info_sufficient = False
        
        print(f"📋 DB 정보 조회 시작: {data.message} (의도: {question_intent}, 우선순위: {intent_priority})")
        
        # 1. 발의자 정보가 있으면 다른 법안 조회 시도
        if proposer_name:
            print(f"📋 발의자 정보 (DB): {proposer_name}")
            
            # 발의자 이름으로 다른 법안 조회 (다른 법안 질문이거나 발의자 관련 질문인 경우)
            if is_other_bills_question or is_proposer_question:
                try:
                    with engine.connect() as conn:
                        other_query = text("""
                            SELECT bill_name, propose_dt 
                            FROM public.final_training_data_copy_sample10_md 
                            WHERE proposer_name = :proposer_name 
                              AND bill_name != :current_bill_name
                            ORDER BY propose_dt DESC
                            LIMIT 5
                        """)
                        other_df = pd.read_sql(
                            other_query, 
                            conn, 
                            params={"proposer_name": proposer_name, "current_bill_name": data.bill_name}
                        )
                        if not other_df.empty:
                            other_bills = [
                                f"- {row['bill_name']} ({str(row['propose_dt'])[:10] if pd.notna(row['propose_dt']) else ''})"
                                for _, row in other_df.iterrows()
                            ]
                            print(f"📋 발의한 다른 법안 {len(other_bills)}개 발견 (DB)")
                            if is_other_bills_question and other_bills:
                                db_info_sufficient = True  # 다른 법안 질문에 대해 DB 정보 충분
                except Exception as e:
                    print(f"⚠️ 다른 법안 조회 실패: {e}")
        
        # 2. 뉴스 관련 질문이면 DB에서 뉴스 정보 조회
        if is_news_question:
            try:
                with engine.connect() as conn:
                    news_query = text("""
                        SELECT news 
                        FROM public.final_training_data_copy_sample10_md 
                        WHERE bill_name = :bill_name 
                        LIMIT 1
                    """)
                    news_df = pd.read_sql(news_query, conn, params={"bill_name": data.bill_name})
                    if not news_df.empty:
                        news_data = news_df.iloc[0].get('news')
                        if news_data:
                            import json
                            try:
                                if isinstance(news_data, str):
                                    news_list = json.loads(news_data)
                                else:
                                    news_list = news_data
                                
                                if isinstance(news_list, list) and len(news_list) > 0:
                                    db_news = [
                                        f"- {item.get('title', '')}: {item.get('summary', '')[:150]}"
                                        for item in news_list[:3]
                                    ]
                                    print(f"📋 관련 뉴스 {len(db_news)}개 발견 (DB)")
                                    if db_news:
                                        db_info_sufficient = True  # 뉴스 질문에 대해 DB 정보 충분
                            except:
                                pass
            except Exception as e:
                print(f"⚠️ 뉴스 조회 실패: {e}")
        
        # 3. DB 정보로 답변 가능한지 판단 (개선된 분류 시스템 기반)
        # - 확률/예측 질문: DB 정보 충분
        # - 발의자 정보 질문: proposer_name 있으면 충분
        # - 다른 법안 질문: other_bills 있으면 충분
        # - 뉴스 질문: db_news 있으면 충분
        # - 내용 질문: DB의 summary/report_md로 충분
        # - 그 외: DB 정보로 답변 시도, 부족하면 웹 검색
        
        if is_probability_question:
            db_info_sufficient = True  # 확률은 항상 DB에서 조회 가능
        elif is_proposer_question and proposer_name:
            db_info_sufficient = True
        elif is_other_bills_question and other_bills:
            db_info_sufficient = True
        elif is_news_question and db_news:
            db_info_sufficient = True
        elif is_content_question:
            db_info_sufficient = True  # 법안 내용은 DB의 summary/report_md로 충분
        elif is_status_question:
            # 처리 현황은 DB에 없을 수 있으므로 웹 검색 필요
            db_info_sufficient = False
        
        # 분류 시스템의 needs_web_search와 DB 정보 충분성을 결합
        # DB 정보가 충분하면 웹 검색 불필요, 부족하면 분류 시스템의 판단 사용
        if db_info_sufficient:
            needs_web_search = False
        else:
            needs_web_search = intent_result.get('needs_web_search', True)
        
        # DB 정보 수집 (프롬프트에 포함)
        proposer_info = ""
        if proposer_name:
            proposer_info = f"- 발의자: {proposer_name}\n"
        if other_bills:
            proposer_info += f"- 발의한 다른 법안 (DB):\n" + "\n".join(other_bills) + "\n"
        if db_news:
            proposer_info += f"- 관련 뉴스 (DB):\n" + "\n".join(db_news) + "\n"
        
        # RAG 검색 수행 (법안 문서에서 관련 컨텍스트 검색)
        rag_context = ""
        rag_docs = retrieve_relevant_context(
            question=data.message,
            bill_name=data.bill_name,
            engine=engine,
            top_k=3
        )
        
        if rag_docs:
            # RAG 검색 결과를 구조화된 형식으로 변환
            rag_texts = []
            for i, doc in enumerate(rag_docs, 1):
                rag_texts.append(f"[{i}] ({doc['type']}, 유사도: {doc['score']:.2f})\n{doc['text'][:300]}")
            rag_context = "\n\n".join(rag_texts)
            print(f"📚 RAG 컨텍스트 준비 완료: {len(rag_docs)}개 문서")
        
        # 웹 검색 수행 (필요한 경우)
        search_summary = ""
        if needs_web_search:
            # 웹 검색 수행 (DB 정보 보완용)
            print(f"🔍 웹 검색 수행 (DB 정보 보완): {data.message}")
            
            # 검색 쿼리 최적화
            if proposer_name and is_proposer_question:
                search_query = f"{proposer_name} {data.message}"
            elif is_news_question:
                search_query = f"{data.bill_name} {proposer_name if proposer_name else ''} {data.message}".strip()
            else:
                search_query = f"{data.bill_name} {data.message}"
            
            search_results = search_web(search_query, max_results=5)
            
            if search_results:
                # 검색 결과 요약
                search_summary = "\n".join([
                    f"- {result.get('title', '')}: {result.get('body', '')[:200]}"
                    for result in search_results[:3]
                ])
        
        # 개선된 프롬프트 생성
        system_prompt = build_enhanced_prompt(
            question=data.message,
            question_intent=question_intent,
            bill_name=data.bill_name,
            display_prob=display_prob,
            pred_value=pred_value,
            proposer_info=proposer_info,
            ctx=ctx,
            search_summary=search_summary,
            has_web_search=bool(search_summary),
            rag_context=rag_context
        )

        # 대화 히스토리 구성 (최근 10개만 사용하여 토큰 제한)
        messages = [{"role": "system", "content": system_prompt}]
        
        # 히스토리 추가 (최근 10개만)
        if data.conversation_history:
            # 최근 10개 대화만 사용 (토큰 제한 방지)
            recent_history = data.conversation_history[-10:]
            for hist in recent_history:
                # role이 "user" 또는 "assistant"인지 검증
                if isinstance(hist, dict) and "role" in hist and "content" in hist:
                    role = hist.get("role", "").lower()
                    if role in ["user", "assistant"]:
                        messages.append({
                            "role": role,
                            "content": str(hist.get("content", ""))
                        })
        
        # 현재 메시지 추가
        messages.append({"role": "user", "content": data.message})
        
        print(f"📝 메시지 개수: {len(messages)} (시스템 프롬프트 + 히스토리 {len(messages)-2}개 + 현재 메시지)")
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,  # 더 일관된 응답을 위해 낮춤
            max_tokens=300  # 간결한 답변을 위해 조정
        )
        
        raw = response.choices[0].message.content
        final_answer = (raw or "").strip()
        
        # 마크다운 문법 제거 (개선)
        import re
        final_answer = re.sub(r'\*\*([^\*]+)\*\*', r'\1', final_answer)  # **볼드** 제거
        final_answer = re.sub(r'\*([^\*\n]+?)\*', r'\1', final_answer)  # *이탤릭* 제거
        final_answer = re.sub(r'#+\s+', '', final_answer)  # # 제목 제거
        final_answer = re.sub(r'`([^`]+)`', r'\1', final_answer)  # `코드` 제거
        final_answer = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', final_answer)  # [링크](url) 제거
        final_answer = re.sub(r'```[\s\S]*?```', '', final_answer)  # 코드 블록 제거
        final_answer = re.sub(r'^\s*[-*+]\s+', '', final_answer, flags=re.MULTILINE)  # 리스트 마커 제거
        final_answer = re.sub(r'\n{3,}', '\n\n', final_answer)  # 연속된 줄바꿈 정리
        final_answer = final_answer.strip()
        
        # 응답 품질 검증 및 개선
        final_answer = validate_and_improve_response(
            response=final_answer,
            question_intent=question_intent,
            display_prob=display_prob,
            pred_value=pred_value,
            proposer_name=proposer_name
        )
        
        if not final_answer:
            final_answer = "죄송합니다. 답변을 생성하는 데 문제가 발생했습니다. 다시 질문해 주세요."
        
        print(f"🤖 AI 답변 ({question_intent}): {final_answer[:100]}...")

        return {"answer": final_answer}

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ 챗봇 에러: {str(e)}")
        return {"answer": "답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."}





   




if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8888)