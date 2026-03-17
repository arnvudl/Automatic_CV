import os
import re
import csv
from datetime import datetime
from time import time

# ==============================================================
# VARIABLES DE CONFIGURATION
# ==============================================================
CV_FOLDER = "Tom\data"
CSV_OUTPUT = "dataset.csv"
LABELS_FILE = os.path.join(CV_FOLDER, "labels.csv") 

TODAY = time.today() if isinstance(time, int) else datetime.today() 

# ==============================================================
# FONCTIONS UTILITAIRES
# ==============================================================
def calculate_age(dob_str):
    if not dob_str: return None
    try:
        dob = datetime.strptime(dob_str.strip(), "%Y-%m-%d")
        age = TODAY.year - dob.year - ((TODAY.month, TODAY.day) < (dob.month, dob.day))
        return age
    except ValueError:
        return None

def extract_country(address_str):
    if not address_str: return None
    parts = [part.strip() for part in address_str.split(',')]
    return parts[-1] if parts else None

def get_sector(target_role):
    if not target_role: return "Other"
    role = target_role.lower()
    if any(w in role for w in ['analyst', 'financial', 'controller', 'accountant', 'audit', 'finance', 'banking']):
        return "Finance"
    if any(w in role for w in ['developer', 'software', 'engineer', 'it', 'data', 'cloud', 'devops', 'architect']):
        return "IT"
    if any(w in role for w in ['operations', 'supply chain', 'logistics', 'production', 'manufacturing', 'industrial']):
        return "Industry"
    if any(w in role for w in ['public', 'government', 'administration', 'policy']):
        return "Public"
    return "Other"

def get_education_level(diploma):
    if not diploma: return 1
    dip_l = diploma.lower()
    if any(w in dip_l for w in ['phd', 'doctorate', 'doctorat']): return 4
    if any(w in dip_l for w in ['master', 'msc', 'mba', 'm2', 'msc']): return 3
    if any(w in dip_l for w in ['bachelor', 'bsc', 'licence', 'b.sc']): return 2
    return 1

def parse_date(date_str):
    date_str = date_str.strip().lower()
    if date_str == 'present':
        return TODAY
    try:
        return datetime.strptime(date_str, "%Y-%m")
    except ValueError:
        return None

def is_senior(title):
    senior_keywords = ['senior', 'lead', 'manager', 'director', 'head', 'chief', 'principal']
    title_l = title.lower()
    return any(w in title_l for w in senior_keywords)

# ==============================================================
# LOGIQUE PRINCIPALE
# ==============================================================
def parse_cv(filepath, labels_dict):
    filename = os.path.basename(filepath)
    cv_id, _ = os.path.splitext(filename)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extraction des champs simples
    name_m = re.search(r"Name:\s*(.*)", content)
    gender_m = re.search(r"Gender:\s*(.*)", content)
    dob_m = re.search(r"Date of Birth:\s*(.*)", content)
    address_m = re.search(r"Address:\s*(.*)", content)
    email_m = re.search(r"Email:\s*(.*)", content)
    phone_m = re.search(r"Phone:\s*(.*)", content)
    target_role_m = re.search(r"Target Role:\s*(.*)", content)

    target_role = target_role_m.group(1).strip() if target_role_m else None
    
    # Label extraction (depuis le nom ou le contenu)
    label_val = None
    if label_val is None:
        if "invite" in filename.lower(): label_val = 1
        elif "reject" in filename.lower(): label_val = 0
    if label_val is None:
        status_m = re.search(r"(?:Status|Decision):\s*(.*)", content, flags=re.IGNORECASE)
        if status_m:
            s_val = status_m.group(1).lower()
            if "invite" in s_val: label_val = 1
            elif "reject" in s_val: label_val = 0
    if label_val is None and cv_id in labels_dict:
        label_str = str(labels_dict[cv_id]).lower()
        if "invite" in label_str or label_str == "1": label_val = 1
        elif "reject" in label_str or label_str == "0": label_val = 0

    # Sections
    sections = re.split(r"(Education|Experience|Skills|Languages|Certifications):", content)
    sections_dict = {sections[i]: sections[i+1].strip() for i in range(1, len(sections), 2)}

    # Education
    education_level = None
    education_field = None
    if "Education" in sections_dict:
        lines = [l for l in sections_dict["Education"].split('\n') if l.strip()]
        if lines:
            parts = [p.strip() for p in lines[0].split('—')]
            diploma = parts[0] if len(parts) > 0 else None
            education_field = parts[1] if len(parts) > 1 else None
            education_level = get_education_level(diploma)

    # Experience
    nb_jobs = 0
    years_experience = 0.0
    jobs = []
    
    if "Experience" in sections_dict:
        exp_text = sections_dict["Experience"]
        # Match lignes de type: [Poste] — [Entreprise] — [Lieu] — [Date début] to [Date fin]
        job_pattern = r"(?P<title>.+?)\s*—\s*(?P<company>.+?)\s*—\s*(?P<location>.+?)\s*—\s*(?P<start>[A-Za-z0-9\-]+)\s+to\s+(?P<end>[A-Za-z0-9\-]+)"
        for match in re.finditer(job_pattern, exp_text):
            nb_jobs += 1
            start_d = parse_date(match.group('start'))
            end_d = parse_date(match.group('end'))
            if start_d and end_d:
                diff_years = (end_d - start_d).days / 365.25
                if diff_years > 0:
                    years_experience += diff_years
            jobs.append({
                'title': match.group('title').strip(),
                'company': match.group('company').strip()
            })
    
    years_experience = round(years_experience, 1) if years_experience > 0 else 0.0
    avg_job_duration = round(years_experience / nb_jobs, 1) if nb_jobs > 0 else 0.0

    career_progression = 0
    if nb_jobs >= 2:
        first_job = jobs[-1] # En supposant chronologie inversée (le dernier dans le CV est le premier temporellement)
        last_job = jobs[0]   # Le plus récent
        
        # Vérif si changement d'entreprise
        changed_company = len(set([j['company'] for j in jobs])) > 1
        if changed_company and is_senior(last_job['title']) and not is_senior(first_job['title']):
            career_progression = 1

    # Skills
    nb_tech, nb_meth, nb_man = 0, 0, 0
    if "Skills" in sections_dict:
        sk_text = sections_dict["Skills"]
        tech_m = re.search(r"Technical:\s*(.*)", sk_text)
        meth_m = re.search(r"Methods:\s*(.*)", sk_text)
        man_m = re.search(r"Management:\s*(.*)", sk_text)
        
        if tech_m and tech_m.group(1).strip(): nb_tech = len(tech_m.group(1).split(','))
        if meth_m and meth_m.group(1).strip(): nb_meth = len(meth_m.group(1).split(','))
        if man_m and man_m.group(1).strip(): nb_man = len(man_m.group(1).split(','))
    
    total_skills = nb_tech + nb_meth + nb_man

    # Languages
    nb_languages = 0
    has_english = 0
    english_level = 0
    has_french = 0
    has_german = 0
    has_luxembourgish = 0
    
    level_map = {'a1': 1, 'a2': 2, 'b1': 3, 'b2': 4, 'c1': 5, 'c2': 6}

    if "Languages" in sections_dict:
        lang_lines = [l for l in sections_dict["Languages"].split('\n') if l.strip()]
        for line in lang_lines:
            parts = [p.strip() for p in line.split('—')]
            if len(parts) >= 1:
                nb_languages += 1
                lang = parts[0].lower()
                lvl = parts[1].strip().lower() if len(parts) > 1 else ""
                
                if any(k in lang for k in ['english', 'anglais', 'en']):
                    has_english = 1
                    english_level = level_map.get(lvl[:2], 0)
                if any(k in lang for k in ['french', 'français', 'francais', 'fr']):
                    has_french = 1
                if any(k in lang for k in ['german', 'deutsch', 'allemand', 'de']):
                    has_german = 1
                if any(k in lang for k in ['luxembourgish', 'luxembourgeois', 'lëtzebuergesch', 'lu']):
                    has_luxembourgish = 1

    # Certifications
    nb_certifications = 0
    if "Certifications" in sections_dict:
        cert_lines = [l for l in sections_dict["Certifications"].split('\n') if l.strip()]
        nb_certifications = len(cert_lines)
        
    return {
        'cv_id': cv_id,
        'age': calculate_age(dob_m.group(1)) if dob_m else None,
        'gender': gender_m.group(1).strip() if gender_m else None,
        'country': extract_country(address_m.group(1)) if address_m else None,
        'target_role': target_role,
        'sector': get_sector(target_role),
        'education_level': education_level,
        'education_field': education_field,
        'nb_jobs': nb_jobs,
        'years_experience': years_experience,
        'avg_job_duration': avg_job_duration,
        'career_progression': career_progression,
        'nb_technical_skills': nb_tech,
        'nb_methods_skills': nb_meth,
        'nb_management_skills': nb_man,
        'total_skills': total_skills,
        'nb_languages': nb_languages,
        'has_english': has_english,
        'english_level': english_level,
        'has_french': has_french,
        'has_german': has_german,
        'has_luxembourgish': has_luxembourgish,
        'nb_certifications': nb_certifications,
        'label': label_val
    }

def main():
    # Détecter les labels externes (si existants)
    labels_dict = {}
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'cv_id' in row and 'label' in row:
                    labels_dict[row['cv_id']] = row['label']

    if not os.path.exists(CV_FOLDER):
        print(f"Le dossier spécifié n'existe pas : {CV_FOLDER}")
        print("Veuillez créer le dossier et y placer les fichiers .txt")
        return

    # Parcourir et parser
    data_list = []
    labels_trouves = 0

    for filename in os.listdir(CV_FOLDER):
        if filename.endswith(".txt"):
            filepath = os.path.join(CV_FOLDER, filename)
            row = parse_cv(filepath, labels_dict)
            data_list.append(row)
            if row['label'] is not None:
                labels_trouves += 1

    if not data_list:
        print("Aucun CV traité. Vérifiez le dossier CV_FOLDER.")
        return

    # Export CSV
    headers = [
        "cv_id", "age", "gender", "country", "target_role", "sector",
        "education_level", "education_field", "nb_jobs", "years_experience", 
        "avg_job_duration", "career_progression", "nb_technical_skills", 
        "nb_methods_skills", "nb_management_skills", "total_skills",
        "nb_languages", "has_english", "english_level", "has_french", 
        "has_german", "has_luxembourgish", "nb_certifications", "label"
    ]

    filepath_output = os.path.join(os.path.dirname(__file__), CSV_OUTPUT)
    with open(filepath_output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data_list)

    # Affichage récapitulatif
    print(f"Traitement terminé. {len(data_list)} CV(s) traité(s).")
    print(f"Nombre de labels trouvés : {labels_trouves}")
    print(f"Fichier exporté vers : {filepath_output}\n")
    
    print("Aperçu des 5 premières lignes :")
    print(f"{', '.join(headers[:5])}...") # Entête simple
    for index, d in enumerate(data_list[:5]):
        print(f"{d['cv_id']}, {d['age']}, {d['gender']}, {d['country']}, {d['target_role']}...")

if __name__ == "__main__":
    main()