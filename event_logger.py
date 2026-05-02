from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional
import pandas as pd
import csv
import os
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

@dataclass
class ViolationEvent:
    timestamp: str
    source: str
    track_id: str
    person_count: int
    missing_items: str
    screenshot_path: str
    confidence: float
    bbox: str

class EventLogger:
    def __init__(self):
        self.events: List[ViolationEvent] = []

    def add_event(self, event: ViolationEvent):
        self.events.append(event)

    def clear_events(self):
        self.events.clear()

    def to_dataframe(self):
        if not self.events:
            return pd.DataFrame(columns=["timestamp", "source", "track_id", "person_count", "missing_items", "screenshot_path", "confidence", "bbox"])
        return pd.DataFrame([asdict(e) for e in self.events])

    def export_csv(self, filepath: str):
        df = self.to_dataframe()
        # 使用 utf-8-sig 確保 Excel 開啟不亂碼
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        return filepath

    def export_excel(self, filepath: str, stats: dict):
        df = self.to_dataframe()
        wb = Workbook()
        
        # Sheet 1: 事件明細
        ws1 = wb.active
        ws1.title = "Violation Details"
        for r in dataframe_to_rows(df, index=False, header=True):
            ws1.append(r)
        
        # 調整欄寬
        for column in ws1.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = (max_length + 2)
            ws1.column_dimensions[column_letter].width = adjusted_width

        # Sheet 2: 統計摘要
        ws2 = wb.create_sheet(title="Summary")
        ws2.append(["Category", "Count"])
        ws2.append(["Total Violation Events", len(self.events)])
        for item, count in stats.get("missing_counts", {}).items():
            ws2.append([f"{item.capitalize()} Violations", count])
        
        wb.save(filepath)
        return filepath

    def export_pdf(self, filepath: str, stats: dict):
        doc = SimpleDocTemplate(filepath, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []

        # 標題
        elements.append(Paragraph("PPE Violation Report", styles['Title']))
        elements.append(Spacer(1, 12))
        
        # 基本資訊
        elements.append(Paragraph(f"Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        elements.append(Paragraph(f"Total Events: {len(self.events)}", styles['Normal']))
        elements.append(Spacer(1, 12))

        # 統計摘要表格
        summary_data = [["Category", "Count"]]
        for item, count in stats.get("missing_counts", {}).items():
            summary_data.append([f"{item.capitalize()} Violations", str(count)])
        
        summary_table = Table(summary_data, colWidths=[200, 100])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(Paragraph("Summary Statistics", styles['Heading2']))
        elements.append(summary_table)
        elements.append(Spacer(1, 24))

        # 事件明細表格 (只取前 50 筆避免 PDF 過大)
        elements.append(Paragraph("Event Details (Top 50)", styles['Heading2']))
        header = ["Time", "ID", "Missing", "Conf"]
        data = [header]
        for e in self.events[:50]:
            data.append([
                e.timestamp.split(" ")[1], # 只顯示時間部分
                e.track_id,
                e.missing_items,
                f"{e.confidence:.2f}"
            ])
        
        detail_table = Table(data, colWidths=[80, 80, 150, 60])
        detail_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 8)
        ]))
        elements.append(detail_table)
        
        # 截圖展示 (只放最近 5 筆)
        if self.events:
            elements.append(Spacer(1, 24))
            elements.append(Paragraph("Recent Violation Screenshots", styles['Heading2']))
            for e in self.events[-5:]:
                if e.screenshot_path and os.path.exists(e.screenshot_path):
                    try:
                        img = Image(e.screenshot_path, width=400, height=250)
                        elements.append(Paragraph(f"Time: {e.timestamp} | ID: {e.track_id}", styles['Normal']))
                        elements.append(img)
                        elements.append(Spacer(1, 12))
                    except:
                        pass

        doc.build(elements)
        return filepath
