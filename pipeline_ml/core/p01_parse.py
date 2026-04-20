"""
p01_parse.py - CV Parsing
"""

import os
import re
import csv
import uuid
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# ==============================================================
# CONFIG
# ==============================================================
ROOT = Path(__file__).parent.parent.parent
RAW_FOLDER = ROOT / "data" / "raw"
PROCESSED_FOLDER = ROOT / "data" / "processed"
LABELS_FILE = ROOT / "data" / "raw" / "labels.csv"

TODAY = datetime.today()
MAX_WORKERS = min(8, (os.cpu_count() or 1) * 2)

LEVEL_MAP = {'a1': 1, 'a2': 2, 'b1': 3, 'b2': 4, 'c1': 5, 'c2': 6}
SENIOR_KW = {'senior', 'lead', 'manager', 'director', 'head', 'chief', 'principal'}

RE_NAME         = re.compile(r"Name:\s*(.*)", re.IGNORECASE)
RE_GENDER       = re.compile(r"Gender:\s*(.*)", re.IGNORECASE)
RE_DOB          = re.compile(r"Date of Birth:\s*(.*)", re.IGNORECASE)
RE_EMAIL        = re.compile(r"Email:\s*(.*)", re.IGNORECASE)
RE_EMAIL_MD     = re.compile(r"\[([^\]]+)\]\(mailto:[^\)]+\)")
RE_PHONE        = re.compile(r"Phone:\s*(.*)", re.IGNORECASE)
RE_TARGET_ROLE  = re.compile(r"Target Role:\s*(.*)", re.IGNORECASE)
RE_STATUS       = re.compile(r"(?:Status|Decision):\s*(.*)", re.IGNORECASE)
RE_SECTIONS     = re.compile(r"(Education|Experience|Skills|Languages|Certifications):", re.IGNORECASE)
_SEP = r"(?:\s*[-—]\s*)"   # tiret court OU tiret long (em dash)
RE_JOB          = re.compile(
    r"(?P<title>.+?)" + _SEP + r"(?P<company>.+?)" + _SEP + r".+?" + _SEP +
    r"(?P<start>[A-Za-z0-9\-]+)\s+to\s+(?P<end>[A-Za-z0-9\-]+)"
)
RE_TECH         = re.compile(r"Technical:\s*(.*)", re.IGNORECASE)
RE_METH         = re.compile(r"Methods:\s*(.*)", re.IGNORECASE)
RE_MAN          = re.compile(r"Management:\s*(.*)", re.IGNORECASE)

def _calculate_age(dob_str: str) -> Optional[int]:
    try:
        dob = datetime.strptime(dob_str.strip(), "%Y-%m-%d")
        return TODAY.year - dob.year - ((TODAY.month, TODAY.day) < (dob.month, dob.day))
    except (ValueError, AttributeError):
        return None

def _get_sector(target_role: Optional[str]) -> str:
    if not target_role:
        return "Other"
    role = target_role.lower()
    if any(w in role for w in ('analyst', 'financial', 'controller', 'accountant', 'audit', 'finance', 'banking')):
        return "Finance"
    if any(w in role for w in ('developer', 'software', 'engineer', 'it', 'data', 'cloud', 'devops', 'architect')):
        return "IT"
    if any(w in role for w in ('operations', 'supply chain', 'logistics', 'production', 'manufacturing', 'industrial')):
        return "Industry"
    if any(w in role for w in ('public', 'government', 'administration', 'policy')):
        return "Public"
    return "Other"

def _get_education_level(diploma: Optional[str]) -> int:
    if not diploma:
        return 1
    d = diploma.lower()
    if any(w in d for w in ('phd', 'doctorate', 'doctorat')):
        return 4
    if any(w in d for w in ('master', 'msc', 'mba', 'm2')):
        return 3
    if any(w in d for w in ('bachelor', 'bsc', 'licence', 'b.sc')):
        return 2
    return 1

def _parse_date(date_str: str) -> Optional[datetime]:
    s = date_str.strip().lower()
    if s == 'present':
        return TODAY
    try:
        return datetime.strptime(s, "%Y-%m")
    except ValueError:
        return None

def _is_senior(title: str) -> bool:
    return bool(SENIOR_KW.intersection(title.lower().split()))

def _split_sections(content: str) -> dict:
    parts = RE_SECTIONS.split(content)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}

def parse_cv(filepath: Path, labels_dict: dict) -> tuple:
    content = filepath.read_text(encoding='utf-8')
    filename = filepath.stem

    name_m    = RE_NAME.search(content)
    gender_m  = RE_GENDER.search(content)
    dob_m     = RE_DOB.search(content)
    email_m   = RE_EMAIL.search(content)
    phone_m   = RE_PHONE.search(content)

    name   = name_m.group(1).strip()   if name_m   else None
    gender = gender_m.group(1).strip() if gender_m else None
    dob    = dob_m.group(1).strip()    if dob_m    else None
    if email_m:
        raw_email = email_m.group(1).strip()
        md = RE_EMAIL_MD.search(raw_email)
        email = md.group(1) if md else raw_email
    else:
        email = None
    phone  = phone_m.group(1).strip()  if phone_m  else None
    age    = _calculate_age(dob)

    cv_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, filename))

    label_val = None
    fn_lower = filename.lower()
    if "invite" in fn_lower:
        label_val = 1
    elif "reject" in fn_lower:
        label_val = 0
    if label_val is None:
        status_m = RE_STATUS.search(content)
        if status_m:
            s = status_m.group(1).lower()
            if "invite" in s:
                label_val = 1
            elif "reject" in s:
                label_val = 0
    if label_val is None and filename in labels_dict:
        raw = str(labels_dict[filename]).lower()
        if "invite" in raw or raw == "1":
            label_val = 1
        elif "reject" in raw or raw == "0":
            label_val = 0

    tr_m = RE_TARGET_ROLE.search(content)
    target_role = tr_m.group(1).strip() if tr_m else None
    sector = _get_sector(target_role)
    sections = _split_sections(content)

    education_level = 1
    education_field = None
    if "Education" in sections:
        lines = [line for line in sections["Education"].split('\n') if line.strip()]
        if lines:
            parts = [p.strip() for p in re.split(r'[-—]', lines[0])]
            education_level = _get_education_level(parts[0] if parts else None)
            education_field = parts[1] if len(parts) > 1 else None

    nb_jobs = 0
    years_experience = 0.0
    jobs = []
    if "Experience" in sections:
        for m in RE_JOB.finditer(sections["Experience"]):
            nb_jobs += 1
            start_d = _parse_date(m.group('start'))
            end_d   = _parse_date(m.group('end'))
            if start_d and end_d:
                diff = (end_d - start_d).days / 365.25
                if diff > 0:
                    years_experience += diff
            jobs.append({'title': m.group('title').strip(), 'company': m.group('company').strip()})

    years_experience = round(years_experience, 1)
    avg_job_duration = round(years_experience / nb_jobs, 1) if nb_jobs > 0 else 0.0

    career_progression = 0
    if nb_jobs >= 2:
        first_job, last_job = jobs[-1], jobs[0]
        if (len({j['company'] for j in jobs}) > 1
                and _is_senior(last_job['title'])
                and not _is_senior(first_job['title'])):
            career_progression = 1

    nb_tech = nb_meth = nb_man = 0
    if "Skills" in sections:
        sk = sections["Skills"]
        t = RE_TECH.search(sk)
        m2 = RE_METH.search(sk)
        mn = RE_MAN.search(sk)
        if t  and t.group(1).strip():  nb_tech = len(t.group(1).split(','))
        if m2 and m2.group(1).strip(): nb_meth = len(m2.group(1).split(','))
        if mn and mn.group(1).strip(): nb_man  = len(mn.group(1).split(','))

    total_skills = nb_tech + nb_meth + nb_man

    nb_languages = has_english = has_french = has_german = has_luxembourgish = 0
    english_level = 0
    if "Languages" in sections:
        for line in (ln for ln in sections["Languages"].split('\n') if ln.strip()):
            parts = [p.strip() for p in re.split(r'[-—]', line)]
            nb_languages += 1
            lang = parts[0].lower()
            lvl  = parts[1].lower() if len(parts) > 1 else ""
            if any(k in lang for k in ('english', 'anglais')):
                has_english = 1
                english_level = LEVEL_MAP.get(lvl[:2], 0)
            if any(k in lang for k in ('french', 'français', 'francais')):
                has_french = 1
            if any(k in lang for k in ('german', 'deutsch', 'allemand')):
                has_german = 1
            if any(k in lang for k in ('luxembourgish', 'luxembourgeois', 'lëtzebuergesch')):
                has_luxembourgish = 1

    nb_certifications = 0
    if "Certifications" in sections:
        nb_certifications = len([ln for ln in sections["Certifications"].split('\n') if ln.strip()])

    identity_row = {
        'cv_id':  cv_id,
        'source_filename': filename,
        'name':   name,
        'email':  email,
        'phone':  phone,
        'gender': gender,
        'age':    age,
    }

    if years_experience < 3:
        profile_type = "junior"
    elif years_experience < 8:
        profile_type = "intermediate"
    else:
        profile_type = "senior"

    feature_row = {
        'cv_id':                cv_id,
        'profile_type':         profile_type,
        'target_role':          target_role,
        'sector':               sector,
        'education_level':      education_level,
        'education_field':      education_field,
        'nb_jobs':              nb_jobs,
        'years_experience':     years_experience,
        'avg_job_duration':     avg_job_duration,
        'career_progression':   career_progression,
        'nb_technical_skills':  nb_tech,
        'nb_methods_skills':    nb_meth,
        'nb_management_skills': nb_man,
        'total_skills':         total_skills,
        'nb_languages':         nb_languages,
        'has_english':          has_english,
        'english_level':        english_level,
        'has_french':           has_french,
        'has_german':           has_german,
        'has_luxembourgish':    has_luxembourgish,
        'nb_certifications':    nb_certifications,
        'label':                label_val,
    }

    return identity_row, feature_row

def main():
    if not RAW_FOLDER.exists():
        print(f"Dossier introuvable : {RAW_FOLDER}")
        return

    PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)

    labels_dict = {}
    if LABELS_FILE.exists():
        with LABELS_FILE.open(encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if 'filename' in row and 'passed_next_stage' in row:
                    stem = Path(row['filename']).stem
                    labels_dict[stem] = row['passed_next_stage']

    cv_files = list(RAW_FOLDER.glob("*.txt"))
    if not cv_files:
        print("Aucun fichier .txt trouvé dans", RAW_FOLDER)
        return

    identities = []
    features   = []
    errors = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(parse_cv, f, labels_dict): f for f in cv_files}
        for future in as_completed(futures):
            filepath = futures[future]
            try:
                id_row, feat_row = future.result()
                identities.append(id_row)
                features.append(feat_row)
            except Exception as exc:
                errors += 1
                print(f"  [ERREUR] {filepath.name} : {exc}")

    feat_headers = [
        'cv_id', 'profile_type', 'target_role', 'sector',
        'education_level', 'education_field',
        'nb_jobs', 'years_experience', 'avg_job_duration', 'career_progression',
        'nb_technical_skills', 'nb_methods_skills', 'nb_management_skills', 'total_skills',
        'nb_languages', 'has_english', 'english_level', 'has_french', 'has_german',
        'has_luxembourgish', 'nb_certifications', 'label',
    ]
    features_path = PROCESSED_FOLDER / "features.csv"
    with features_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=feat_headers)
        writer.writeheader()
        writer.writerows(features)

    identities_path = PROCESSED_FOLDER / "identities.csv"
    with identities_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['cv_id', 'source_filename', 'name', 'email', 'phone', 'gender', 'age'])
        writer.writeheader()
        writer.writerows(identities)

    print(f"CV traites : {len(features)} | Fichiers generes dans {PROCESSED_FOLDER}")

if __name__ == "__main__":
    main()
