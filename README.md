# Automated Resume Screening System

## Introduction
The **Automated Resume Screening System** leverages **Natural Language Processing (NLP)** and **Machine Learning (ML)** to efficiently analyze and rank resumes based on job relevance. This project aims to automate the hiring process by reducing manual effort, improving accuracy, and minimizing human bias.

## Features
- **Resume Parsing:** Extracts text from resumes (PDF/DOCX format).
- **Content Analysis:** Uses **TF-IDF vectorization** and **cosine similarity** for resume-job description matching.
- **Skill Extraction:** Identifies key skills and qualifications.
- **Experience Evaluation:** Estimates years of experience using pattern recognition.
- **Ranking System:** Implements a **weighted scoring algorithm** to rank candidates.
- **Report Generation:** Provides structured reports for recruiters.

## Technology Stack
### Programming Languages & Libraries
- **Python** (Primary Language)
- **NLP:** NLTK, SpaCy
- **ML Models:** Scikit-learn, TensorFlow (if deep learning is used)
- **Data Processing:** Pandas, NumPy

### Frameworks & Tools
- **Flask/Django** (For Web Application, if applicable)
- **SQLite/PostgreSQL** (For storing resume data)
- **Apache Tika** (For extracting text from PDF/DOCX files)

## System Architecture
1. **Resume Upload:** Users upload resumes.
2. **Text Extraction:** Extracts relevant information.
3. **Skill & Experience Analysis:** NLP techniques analyze resume content.
4. **Job Matching & Ranking:** ML algorithms rank resumes based on job relevance.
5. **Report Generation:** Displays the ranked results.

## Installation
1. Clone the repository:
   ```sh
   git clone : git clone https://github.com/AaryanPathak31/AI-powered-Resume-Screening-and-Ranking-System-P1-.git
   cd AI-powered-Resume-Screening-and-Ranking-System-P1-
   ```
2. Run the application:
   ```sh
   python app.py
   ```

## Future Work
- Improve contextual understanding of job descriptions.
- Expand ML models for better accuracy.
- Integrate with ATS (Applicant Tracking Systems).

## References
[1] J. Kaur and R. Sharma, “Automated Resume Screening Using Natural Language Processing,” *International Journal of Computer Applications*, Vol. 180, No. 42, 2018.  
[2] M. H. Yang, D. J. Kriegman, and N. Ahuja, “Detecting Faces in Images: A Survey,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, Vol. 24, No. 1, 2002.
