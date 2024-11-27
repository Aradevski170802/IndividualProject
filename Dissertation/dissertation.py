from fpdf import FPDF

class DissertationPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font('Times', 'I', 10)
            self.cell(0, 10, f'Dissertation Paper', align='C', ln=1)
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Times', 'I', 10)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def title_page(self):
        self.add_page()
        self.set_font('Times', '', 14)
        self.cell(0, 10, 'CITY College', ln=True, align='C')
        self.cell(0, 10, 'University of York Europe Campus', ln=True, align='C')
        self.cell(0, 10, 'Computer Science Department', ln=True, align='C')
        self.ln(20)

        self.set_font('Times', 'B', 16)
        self.cell(0, 10, 'UNDERGRADUATE INDIVIDUAL PROJECT', ln=True, align='C')
        self.ln(20)

        self.set_font('Times', 'B', 18)
        self.cell(0, 10, 'Algorithmic Showdown:', ln=True, align='C')
        self.cell(0, 10, 'Unveiling the Best in Classification', ln=True, align='C')
        self.ln(20)

        self.set_font('Times', '', 12)
        self.multi_cell(
            0, 10,
            "This report is submitted in partial fulfillment of the requirement for the degree of "
            "Bachelors in Computer Science with Honours by", align='C'
        )
        self.ln(10)

        self.set_font('Times', 'B', 14)
        self.cell(0, 10, 'Aleksandar Radevski', ln=True, align='C')
        self.ln(20)

        self.set_font('Times', '', 12)
        self.cell(0, 10, 'May 2024', ln=True, align='C')
        self.ln(10)

        self.cell(0, 10, 'Approved', ln=True, align='C')
        self.ln(10)

        self.set_font('Times', 'B', 12)
        self.cell(0, 10, 'Dr. Ourania Mangira', ln=True, align='C')
        self.ln(20)

        self.set_draw_color(0, 0, 0)
        self.line(60, self.get_y() + 10, 150, self.get_y() + 10)
        self.line(60, self.get_y() + 20, 150, self.get_y() + 20)

    def table_of_contents(self):
        self.add_page()
        self.set_font('Times', 'B', 16)
        self.cell(0, 10, 'Contents', ln=True, align='C')
        self.ln(10)

        self.set_font('Times', '', 12)
        contents = [
            ("1 Introduction", 1),
            ("      1.1 Aim and Objectives", 2),
            ("      1.2 Report Structure", 3),
            ("2 Literature Review", 5),
            ("      2.1 Brief Historical Overview", 6),
            ("      2.2 Importance of HR Metrics", 10),
            ("      2.3 Data Analysis Techniques", 14),
            ("3 Methodology", 18),
            ("4 Results and Analysis", 25),
            ("5 Discussion", 30),
            ("6 Conclusion", 34),
            ("References", 37),
            ("Appendices", 40),
        ]

        for section, page in contents:
            section_width = self.get_string_width(section)
            page_width = self.get_string_width(str(page))
            
            # Define total width for the table of contents (3/5 of the page width)
            toc_width = self.w * 3 / 5
            margin_width = (self.w - toc_width) / 2  # Left and right margins (1/5 each)

            # Calculate the space for dots
            dots_width = toc_width - (section_width + page_width)  # Space left for dots
            dots = "." * (int(dots_width / self.get_string_width(".")))  # Adjust dot count

            # Set the x-position to start at the margin
            self.set_x(margin_width)
            self.cell(toc_width, 10, f"{section} {dots} {page}", ln=True)

    def introduction(self):
        self.add_page()
        self.set_font('Times', 'B', 16)
        self.cell(0, 10, '1. Introduction', ln=True, align='C')
        self.ln(10)
    
        # Set margins for left and right spacing
        self.set_left_margin(26)  # Adjust the left margin
        self.set_right_margin(26)  # Adjust the right margin
    
        self.set_font('Times', '', 12)
        intro_text = (
            "Nowadays, classification algorithms are an indispensable part of machine learning. They help solve complex "
            "real-world problems. From predicting customer churn to diagnosing or even predicting the possibility of "
            "the existence of a disease, these types of algorithms play a major role in the machine learning industry. "
            "But in order for them to be at a high level, the datasets used for training and evaluation must have "
            "precisely defined characteristics and a clearly defined evaluation value.\n\n"
            "A base or even a key point in evaluating and comparing classification algorithms is the reading of datasets "
            "themselves. Many of these datasets are either of limited scope or have interference that drastically affects "
            "the final results. For the purpose of this coursework, an artificial (synthetic) dataset was created that is "
            "easy to read and at the same time satisfies the requirements for easy and at the same time conscious "
            "comparison of different classification algorithms.\n\n"
            "This dissertation focuses on two main goals: first, to see, compare and describe the efficiency, accuracy, and "
            "last but not least the ease of writing of different classification algorithms; and second, to demonstrate the "
            "importance of data quality and design in achieving satisfactory results. Specifically, this paper examines the "
            "performance of algorithms on balanced versus unbalanced data and complete versus incomplete data sets."
        )
        self.multi_cell(0, 8, intro_text, align='J')
    
        # Reset margins to default for other sections
        self.set_left_margin(10)
        self.set_right_margin(10)


# Create the dissertation PDF
pdf = DissertationPDF()
pdf.set_auto_page_break(auto=True, margin=15)

# Add sections
pdf.title_page()
pdf.table_of_contents()
pdf.introduction()

# Save the PDF
output_pdf = "Dissertation.pdf"
pdf.output(output_pdf)

print(f"The dissertation sample with a detailed table of contents has been saved as {output_pdf}.")
