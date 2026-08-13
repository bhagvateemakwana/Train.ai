from io import BytesIO
import json
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.graphics import renderPDF
from PIL import Image as PILImage
import os
import textwrap

class NumberedCanvas:
    def __init__(self, canvas, doc):
        self.canvas = canvas
        self.doc = doc

    def draw_page_number(self):
        page_num = self.canvas.getPageNumber()
        text = f"Page {page_num}"
        self.canvas.setFont("Helvetica", 9)
        self.canvas.setFillColor(colors.HexColor('#7f8c8d'))
        self.canvas.drawRightString(A4[0] - 72, 30, text)

    def draw_header_line(self):
        if self.canvas.getPageNumber() > 1:
            self.canvas.setStrokeColor(colors.HexColor('#3498db'))
            self.canvas.setLineWidth(2)
            self.canvas.line(72, A4[1] - 50, A4[0] - 72, A4[1] - 50)

class ModelReportGenerator:
    def __init__(self, trained_model):
        self.model = trained_model
        self.buffer = BytesIO()
        self.doc = SimpleDocTemplate(
            self.buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=80,
            bottomMargin=50,
            title=f"ML Model Report - {self.model.model_name}",
            author="ML Analysis System"
        )
        self.styles = getSampleStyleSheet()
        self.story = []
        
        self.primary_color = colors.HexColor('#2c3e50')
        self.secondary_color = colors.HexColor('#3498db')
        self.accent_color = colors.HexColor('#e74c3c')
        self.success_color = colors.HexColor('#27ae60')
        self.light_bg = colors.HexColor('#f8f9fa')
        self.medium_bg = colors.HexColor('#ecf0f1')
        self.text_light = colors.HexColor('#7f8c8d')
        
        self._setup_styles()

    def _setup_styles(self):
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=28,
            spaceAfter=20,
            spaceBefore=20,
            alignment=TA_CENTER,
            textColor=self.primary_color,
            fontName='Helvetica-Bold',
            leading=34
        )
        
        self.subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=self.styles['Normal'],
            fontSize=16,
            alignment=TA_CENTER,
            spaceAfter=15,
            textColor=self.secondary_color,
            fontName='Helvetica',
            leading=20
        )
        
        self.date_style = ParagraphStyle(
            'DateStyle',
            parent=self.styles['Normal'],
            fontSize=11,
            alignment=TA_CENTER,
            spaceAfter=40,
            textColor=self.text_light,
            fontName='Helvetica-Oblique'
        )
        
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=18,
            spaceAfter=15,
            spaceBefore=25,
            textColor=self.primary_color,
            fontName='Helvetica-Bold',
            borderWidth=0,
            borderPadding=0,
            leftIndent=0,
            bulletIndent=0
        )
        
        self.subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=18,
            textColor=self.secondary_color,
            fontName='Helvetica-Bold'
        )
        
        self.body_style = ParagraphStyle(
            'CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            textColor=colors.black,
            fontName='Helvetica',
            alignment=TA_JUSTIFY,
            leading=14
        )
        
        self.caption_style = ParagraphStyle(
            'Caption',
            parent=self.styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            textColor=self.text_light,
            fontName='Helvetica-Oblique',
            spaceAfter=15
        )
        
        # New style for table cell content
        self.table_cell_style = ParagraphStyle(
            'TableCell',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.black,
            fontName='Helvetica',
            alignment=TA_LEFT,
            leading=12,
            wordWrap='LTR',
            leftIndent=0,
            rightIndent=0,
            spaceAfter=0,
            spaceBefore=0
        )

    def _format_features_for_table(self, features_data, max_width=250):
        """Format features list for table display with proper wrapping"""
        try:
            if isinstance(features_data, str):
                features = json.loads(features_data)
            else:
                features = features_data
            
            if not isinstance(features, list):
                return str(features)[:max_width] + '...' if len(str(features)) > max_width else str(features)
            
            # Create formatted feature list
            features_text = ""
            current_line_length = 0
            line_limit = 50  # Characters per line
            
            for i, feature in enumerate(features):
                feature_str = str(feature)
                
                # Add feature with proper line breaks
                if i == 0:
                    features_text = feature_str
                    current_line_length = len(feature_str)
                else:
                    separator = ", "
                    if current_line_length + len(separator) + len(feature_str) > line_limit:
                        features_text += ",<br/>" + feature_str
                        current_line_length = len(feature_str)
                    else:
                        features_text += separator + feature_str
                        current_line_length += len(separator) + len(feature_str)
                
                # If we have too many features, truncate
                if len(features_text) > max_width:
                    remaining_count = len(features) - i
                    if remaining_count > 0:
                        features_text += f"<br/>... and {remaining_count} more features"
                    break
            
            return features_text
            
        except Exception as e:
            # Fallback for any parsing errors
            features_str = str(features_data)
            if len(features_str) > max_width:
                # Use textwrap to break long lines
                wrapped = textwrap.fill(features_str, width=50)
                lines = wrapped.split('\n')
                if len('\n'.join(lines)) > max_width:
                    # If still too long, truncate
                    truncated = features_str[:max_width-20]
                    return truncated + '...'
                return '<br/>'.join(lines)
            return features_str

    def generate_report(self):
        self._add_cover_page()
        self._add_executive_summary()
        self._add_model_info()
        self._add_model_stats()
        self._add_model_graphs()
        self._add_conclusion()
        
        def on_first_page(canvas, doc):
            pass
            
        def on_later_pages(canvas, doc):
            numbered_canvas = NumberedCanvas(canvas, doc)
            numbered_canvas.draw_page_number()
            numbered_canvas.draw_header_line()
        
        self.doc.build(
            self.story,
            onFirstPage=on_first_page,
            onLaterPages=on_later_pages
        )
        
        pdf = self.buffer.getvalue()
        self.buffer.close()
        return pdf

    def _add_cover_page(self):
        title = "MACHINE LEARNING MODEL REPORT"
        self.story.append(Paragraph(title, self.title_style))
        self.story.append(Spacer(1, 30))
        
        model_name_style = ParagraphStyle(
            'ModelNameStyle',
            parent=self.subtitle_style,
            fontSize=20,
            textColor=colors.white,
            backColor=self.secondary_color,
            borderWidth=1,
            borderColor=self.secondary_color,
            borderPadding=15,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        model_name = f"{self.model.model_name}"
        self.story.append(Paragraph(model_name, model_name_style))
        self.story.append(Spacer(1, 40))
        
        model_type_style = ParagraphStyle(
            'ModelTypeStyle',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=self.primary_color,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            borderWidth=2,
            borderColor=self.primary_color,
            borderPadding=10,
            spaceAfter=30
        )
        
        model_type_text = f"Model Type: {self.model.get_model_type_display()}"
        self.story.append(Paragraph(model_type_text, model_type_style))
        
        date_text = f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
        self.story.append(Paragraph(date_text, self.date_style))
        
        self.story.append(Spacer(1, 100))
        
        footer_text = "Automated Machine Learning Analysis System"
        footer_style = ParagraphStyle(
            'CoverFooter',
            parent=self.styles['Normal'],
            fontSize=12,
            alignment=TA_CENTER,
            textColor=self.text_light,
            fontName='Helvetica-Oblique'
        )
        self.story.append(Paragraph(footer_text, footer_style))
        
        self.story.append(PageBreak())

    def _add_executive_summary(self):
        self.story.append(Paragraph("Executive Summary", self.heading_style))
        
        summary_points = []
        
        if hasattr(self.model, 'stats') and self.model.stats:
            stats = self.model.stats
            if stats.accuracy:
                summary_points.append(f"Model achieves {stats.accuracy:.1%} accuracy on the test dataset")
            if stats.f1_score:
                summary_points.append(f"F1-Score of {stats.f1_score:.3f} indicates balanced precision and recall")
            if stats.r2_score:
                summary_points.append(f"R² Score of {stats.r2_score:.3f} explains the variance in the target variable")
        
        summary_points.extend([
            f"Model was trained on {datetime.now().strftime('%B %d, %Y')} using {self.model.get_model_type_display()} algorithm",
            f"Target variable: {self.model.target_column}",
            "Comprehensive visualizations and metrics included for model evaluation"
        ])
        
        summary_text = "This report provides a comprehensive analysis of the machine learning model performance. Key findings include:"
        self.story.append(Paragraph(summary_text, self.body_style))
        self.story.append(Spacer(1, 12))
        
        for point in summary_points:
            bullet_style = ParagraphStyle(
                'BulletStyle',
                parent=self.body_style,
                leftIndent=20,
                bulletIndent=10,
                spaceAfter=6
            )
            self.story.append(Paragraph(f"• {point}", bullet_style))
        
        self.story.append(Spacer(1, 25))

    def _add_model_info(self):
        self.story.append(Paragraph("Model Configuration", self.heading_style))
        
        model_data = [
            ['Parameter', 'Value'],
            ['Model Algorithm', self.model.get_model_type_display()],
            ['Target Variable', self.model.target_column],
            ['Training Date', self.model.created_at.strftime('%B %d, %Y')],
            ['Model Visibility', 'Public' if self.model.is_public else 'Private'],
            ['Community Likes', str(self.model.likes)],
        ]
        
        if self.model.polynomial_degree:
            model_data.append(['Polynomial Degree', str(self.model.polynomial_degree)])
        
        if self.model.features:
            formatted_features = self._format_features_for_table(self.model.features)
            features_paragraph = Paragraph(formatted_features, self.table_cell_style)
            model_data.append(['Input Features', features_paragraph])
        
        table = Table(model_data, colWidths=[2.2*inch, 4.3*inch], repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.primary_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            
            ('BACKGROUND', (0, 1), (0, -1), self.light_bg),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (0, -1), 11),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            
            ('FONTNAME', (1, 1), (1, -1), 'Helvetica'),
            ('FONTSIZE', (1, 1), (1, -1), 10),
            ('ALIGN', (1, 1), (1, -1), 'LEFT'),
            
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('LEFTPADDING', (0, 0), (-1, -1), 15),
            ('RIGHTPADDING', (0, 0), (-1, -1), 15),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7')),
            
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.light_bg]),
        ]))
        
        self.story.append(table)
        self.story.append(Spacer(1, 25))

    def _add_model_stats(self):
        if not hasattr(self.model, 'stats') or not self.model.stats:
            return
        
        stats = self.model.stats
        self.story.append(Paragraph("Performance Metrics", self.heading_style))
        
        interpretation_text = "The following metrics evaluate the model's predictive performance:"
        self.story.append(Paragraph(interpretation_text, self.body_style))
        self.story.append(Spacer(1, 15))
        
        all_stats_data = [['Metric', 'Value', 'Interpretation']]
        
        if any([stats.r2_score, stats.mse, stats.mae]):
            if stats.r2_score is not None:
                r2_interpretation = self._interpret_r2_score(stats.r2_score)
                all_stats_data.append(['R² Score', f"{stats.r2_score:.4f}", r2_interpretation])
            
            if stats.mse is not None:
                all_stats_data.append(['Mean Squared Error', f"{stats.mse:.4f}", "Lower values indicate better fit"])
            
            if stats.mae is not None:
                all_stats_data.append(['Mean Absolute Error', f"{stats.mae:.4f}", "Average prediction error"])
        
        if any([stats.accuracy, stats.precision, stats.recall, stats.f1_score]):
            if stats.accuracy is not None:
                acc_interpretation = self._interpret_accuracy(stats.accuracy)
                all_stats_data.append(['Accuracy', f"{stats.accuracy:.4f} ({stats.accuracy:.1%})", acc_interpretation])
            
            if stats.precision is not None:
                all_stats_data.append(['Precision', f"{stats.precision:.4f} ({stats.precision:.1%})", "True positives / All positive predictions"])
            
            if stats.recall is not None:
                all_stats_data.append(['Recall', f"{stats.recall:.4f} ({stats.recall:.1%})", "True positives / All actual positives"])
            
            if stats.f1_score is not None:
                f1_interpretation = self._interpret_f1_score(stats.f1_score)
                all_stats_data.append(['F1 Score', f"{stats.f1_score:.4f}", f1_interpretation])
        
        if len(all_stats_data) > 1:
            metrics_table = Table(all_stats_data, colWidths=[2*inch, 1.5*inch, 2.5*inch], repeatRows=1)
            metrics_table.setStyle(TableStyle([
                # Header styling
                ('BACKGROUND', (0, 0), (-1, 0), self.success_color),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                
                # Data styling
                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (1, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                ('ALIGN', (1, 1), (1, -1), 'CENTER'),
                ('ALIGN', (2, 1), (2, -1), 'LEFT'),
                
                # General styling
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ('LEFTPADDING', (0, 0), (-1, -1), 12),
                ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.light_bg]),
            ]))
            
            self.story.append(metrics_table)
            self.story.append(Spacer(1, 25))

    def _interpret_r2_score(self, r2):
        if r2 >= 0.9:
            return "Excellent fit"
        elif r2 >= 0.7:
            return "Good fit"
        elif r2 >= 0.5:
            return "Moderate fit"
        else:
            return "Poor fit"

    def _interpret_accuracy(self, acc):
        if acc >= 0.9:
            return "Excellent performance"
        elif acc >= 0.8:
            return "Good performance"
        elif acc >= 0.7:
            return "Fair performance"
        else:
            return "Needs improvement"

    def _interpret_f1_score(self, f1):
        """Interpret F1 score"""
        if f1 >= 0.8:
            return "Well balanced precision/recall"
        elif f1 >= 0.6:
            return "Reasonably balanced"
        else:
            return "Imbalanced precision/recall"

    def _add_model_graphs(self):
        graphs = self.model.graphs.all()
        if not graphs:
            return
        
        self.story.append(Paragraph("Model Visualizations", self.heading_style))
        
        intro_text = "The following visualizations provide insights into the model's performance and behavior:"
        self.story.append(Paragraph(intro_text, self.body_style))
        self.story.append(Spacer(1, 20))
        
        for i, graph in enumerate(graphs):
            graph_elements = []
            
            if graph.title:
                graph_elements.append(Paragraph(f"{i+1}. {graph.title}", self.subheading_style))

            if graph.description:
                desc_style = ParagraphStyle(
                    'GraphDesc',
                    parent=self.body_style,
                    fontSize=11,
                    spaceAfter=12,
                    alignment=TA_JUSTIFY,
                    leftIndent=15
                )
                graph_elements.append(Paragraph(graph.description, desc_style))
            
            if graph.graph_image and os.path.exists(graph.graph_image.path):
                try:
                    img = PILImage.open(graph.graph_image.path)
                    img_width, img_height = img.size
                    
                    max_width = 5.5 * inch
                    max_height = 4 * inch
                    
                    scale = min(max_width / img_width, max_height / img_height)
                    new_width = img_width * scale
                    new_height = img_height * scale
                    
                    graph_img = Image(graph.graph_image.path, width=new_width, height=new_height)
                    
                    img_table = Table([[graph_img]], colWidths=[6*inch])
                    img_table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('TOPPADDING', (0, 0), (-1, -1), 15),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
                        ('LEFTPADDING', (0, 0), (-1, -1), 15),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 15),
                        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
                        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7')),
                    ]))
                    
                    graph_elements.append(img_table)
                    
                    if graph.title:
                        caption_text = f"Figure {i+1}: {graph.title}"
                        graph_elements.append(Paragraph(caption_text, self.caption_style))
                    
                except Exception as e:
                    error_text = f"[Image could not be loaded for: {graph.title or 'Unnamed Graph'}]"
                    error_style = ParagraphStyle(
                        'ErrorStyle',
                        parent=self.body_style,
                        fontSize=10,
                        textColor=self.accent_color,
                        alignment=TA_CENTER,
                        backColor=colors.HexColor('#ffeaa7'),
                        borderPadding=10
                    )
                    graph_elements.append(Paragraph(error_text, error_style))
            
            if graph_elements:
                self.story.append(KeepTogether(graph_elements))
                self.story.append(Spacer(1, 25))

    def _add_conclusion(self):
        self.story.append(Paragraph("Conclusion", self.heading_style))
        
        conclusion_text = f"""
        This automated report provides a comprehensive overview of the {self.model.model_name} model's 
        performance and characteristics. The model utilizes {self.model.get_model_type_display()} 
        algorithm to predict {self.model.target_column}.
        
        Based on the performance metrics and visualizations presented, stakeholders can make informed 
        decisions about model deployment and potential improvements. For additional analysis or 
        model modifications, please refer to the interactive dashboard.
        """
        
        self.story.append(Paragraph(conclusion_text, self.body_style))
        self.story.append(Spacer(1, 30))
        
        footer_text = """
        <b>About This Report:</b><br/>
        This document was automatically generated by the Machine Learning Model Analysis System. 
        The system provides comprehensive model evaluation, visualization, and reporting capabilities 
        for data science teams.<br/><br/>
        
        <i>For technical support or questions about this report, please contact the ML Engineering team.</i>
        """
        
        footer_style = ParagraphStyle(
            'EnhancedFooter',
            parent=self.body_style,
            fontSize=10,
            alignment=TA_JUSTIFY,
            textColor=self.text_light,
            borderWidth=1,
            borderColor=self.secondary_color,
            borderPadding=20,
            backColor=self.light_bg,
            spaceAfter=0
        )
        
        self.story.append(Paragraph(footer_text, footer_style))