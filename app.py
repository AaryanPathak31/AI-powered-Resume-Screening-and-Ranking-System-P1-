import pandas as pd
import numpy as np
import re
import nltk
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
import os
from sklearn.ensemble import RandomForestClassifier

# Download NLTK resources (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def extract_text_from_resume(file_path):
    """Extract text from PDF or DOCX resume files"""
    text = ""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
    
    elif file_extension == '.docx':
        doc = docx.Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    
    return text

def preprocess_text(text):
    """Clean and preprocess the text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters, numbers, and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

def extract_skills(text, skills_list):
    """Extract skills from text based on a predefined skills list"""
    found_skills = []
    for skill in skills_list:
        skill_pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(skill_pattern, text.lower()):
            found_skills.append(skill)
    
    return found_skills

def extract_education(text):
    """Extract education information from resume text"""
    education_keywords = [
        'bachelor', 'master', 'phd', 'doctorate', 'degree', 'bs', 'ms', 'ba', 'ma',
        'b.s.', 'm.s.', 'b.a.', 'm.a.', 'b.tech', 'm.tech', 'mba'
    ]
    
    education_info = []
    text_lower = text.lower()
    
    for keyword in education_keywords:
        pattern = r'\b' + re.escape(keyword) + r'[^.!?]*[.!?]'
        matches = re.findall(pattern, text_lower)
        education_info.extend(matches)
    
    return education_info

def extract_experience(text):
    """Extract work experience information from resume text"""
    # Look for patterns like "X years of experience" or "X+ years"
    experience_patterns = [
        r'\b(\d+)\+?\s+years?\s+(?:of\s+)?experience\b',
        r'\bexperience\s+(?:of\s+)?(\d+)\+?\s+years?\b'
    ]
    
    for pattern in experience_patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            # Return the highest number of years found
            return max([int(years) for years in matches])
    
    return 0  # Default if no experience info found

def rank_resumes(job_description, resume_files, skills_list, weights=None):
    """
    Rank resumes based on similarity to job description and skills match
    
    Parameters:
    - job_description: String containing the job requirements
    - resume_files: List of paths to resume files
    - skills_list: List of relevant skills to look for
    - weights: Dictionary with weights for different ranking factors
    
    Returns:
    - DataFrame with ranked resumes and scores
    """
    if weights is None:
        weights = {
            'content_similarity': 0.4,
            'skills_match': 0.4,
            'experience': 0.2
        }
    
    # Preprocess job description
    processed_jd = preprocess_text(job_description)
    
    results = []
    
    # Process each resume
    for resume_path in resume_files:
        resume_text = extract_text_from_resume(resume_path)
        processed_resume = preprocess_text(resume_text)
        
        # Calculate content similarity using TF-IDF and cosine similarity
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform([processed_jd, processed_resume])
        content_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Extract skills
        found_skills = extract_skills(resume_text, skills_list)
        skills_match_score = len(found_skills) / len(skills_list) if skills_list else 0
        
        # Extract experience
        years_experience = extract_experience(resume_text)
        # Normalize experience score (assuming 10+ years is max)
        experience_score = min(years_experience / 10, 1.0)
        
        # Extract education
        education_info = extract_education(resume_text)
        
        # Calculate weighted score
        total_score = (
            weights['content_similarity'] * content_similarity +
            weights['skills_match'] * skills_match_score +
            weights['experience'] * experience_score
        )
        
        # Store results
        results.append({
            'resume': os.path.basename(resume_path),
            'content_similarity': content_similarity,
            'skills_match_score': skills_match_score,
            'matched_skills': found_skills,
            'years_experience': years_experience,
            'education': education_info,
            'total_score': total_score
        })
    
    # Create DataFrame and sort by total score
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='total_score', ascending=False).reset_index(drop=True)
    
    return results_df

def train_job_category_classifier(training_data, categories):
    """
    Train a classifier to categorize resumes into job categories
    
    Parameters:
    - training_data: DataFrame with resume text and categories
    - categories: List of job categories
    
    Returns:
    - Trained classifier and vectorizer
    """
    # Preprocess training data
    training_data['processed_text'] = training_data['resume_text'].apply(preprocess_text)
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(training_data['processed_text'])
    y = training_data['category']
    
    # Train classifier
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X, y)
    
    return classifier, vectorizer

def classify_resume_category(resume_text, classifier, vectorizer, categories):
    """Classify a resume into a job category"""
    processed_text = preprocess_text(resume_text)
    text_vector = vectorizer.transform([processed_text])
    category_probs = classifier.predict_proba(text_vector)[0]
    
    # Create dictionary of category probabilities
    category_scores = {categories[i]: category_probs[i] for i in range(len(categories))}
    predicted_category = categories[category_probs.argmax()]
    
    return predicted_category, category_scores

def resume_screening_system(job_description, resume_directory, skills_list, 
                           classifier=None, vectorizer=None, categories=None):
    """
    Complete resume screening and ranking system
    
    Parameters:
    - job_description: String containing job requirements
    - resume_directory: Directory containing resume files
    - skills_list: List of relevant skills
    - classifier, vectorizer, categories: Optional ML components for categorization
    
    Returns:
    - DataFrame with ranked and categorized resumes
    """
    # Get all resume files from directory
    resume_files = [
        os.path.join(resume_directory, f) for f in os.listdir(resume_directory)
        if f.endswith(('.pdf', '.docx'))
    ]
    
    # Rank resumes
    ranked_resumes = rank_resumes(job_description, resume_files, skills_list)
    
    # If classifier is provided, categorize resumes
    if classifier and vectorizer and categories:
        categories_list = []
        category_scores_list = []
        
        for resume_path in resume_files:
            resume_text = extract_text_from_resume(resume_path)
            category, category_scores = classify_resume_category(
                resume_text, classifier, vectorizer, categories
            )
            categories_list.append(category)
            category_scores_list.append(category_scores)
        
        ranked_resumes['predicted_category'] = categories_list
        ranked_resumes['category_scores'] = category_scores_list
    
    return ranked_resumes

def generate_resume_report(ranked_resumes, output_file=None):
    """Generate a detailed report of the resume screening results"""
    report = "# Resume Screening Results\n\n"
    
    for i, row in ranked_resumes.iterrows():
        report += f"## {i+1}. {row['resume']} (Score: {row['total_score']:.2f})\n\n"
        report += f"- **Content Match:** {row['content_similarity']:.2f}\n"
        report += f"- **Skills Match:** {row['skills_match_score']:.2f}\n"
        report += f"- **Experience:** {row['years_experience']} years\n"
        report += f"- **Matched Skills:** {', '.join(row['matched_skills'])}\n"
        
        if 'predicted_category' in row:
            report += f"- **Predicted Job Category:** {row['predicted_category']}\n"
        
        report += "\n"
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
    
    return report

# Main execution
if __name__ == "__main__":
    # Define job description
    job_description = """
    We are seeking a skilled Web Developer proficient in JavaScript, C++, ReactJS, and Java.
    The ideal candidate should have strong front-end development skills, experience with
    React-based web applications, and knowledge of server-side programming using Java and C++.
    """

    # Define relevant skills
    skills_list = [
        'javascript', 'cpp', 'c++', 'reactjs', 'react', 'java', 'html', 'css',
        'web development', 'front-end', 'back-end', 'full-stack'
    ]

    # Set the resume directory
    resume_directory = r"D:\faltu"

    try:
        # Run the system
        results = resume_screening_system(
            job_description=job_description,
            resume_directory=resume_directory,
            skills_list=skills_list
        )

        # Generate and print report
        report = generate_resume_report(results)
        print(report)
        
        # Display top candidates
        print("\nTop Candidates:")
        print(results[['resume', 'total_score', 'matched_skills', 'years_experience']].head())
        
        # Save report to file
        output_file = os.path.join(resume_directory, "resume_screening_results.txt")
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\nDetailed report saved to: {output_file}")
        
    except Exception as e:
        print(f"Error running the resume screening system: {e}")
        print("Please ensure you have the correct resume directory and required libraries installed.")
        print("Required libraries: pandas, numpy, nltk, scikit-learn, PyPDF2, python-docx")
