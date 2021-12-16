from typing import Dict


gender_map = {
    'Female': 2.0,
    'Male': 1.0,
    'Other': 0.0
}

relevent_experience_map = {
    'Has relevent experience':  1,
    'No relevent experience':    0
}

enrolled_university_map = {
    'no_enrollment':  0.0,
    'Full time course':    1.0,
    'Part time course':    2.0
}

education_level_map = {
    'Primary School':    0.0,
    'Graduate':    2.0,
    'Masters':    3.0,
    'High School':    1.0,
    'Phd':    4.0
}

major_map = {
    'STEM':    0.0,
    'Business Degree':    1.0,
    'Arts':    2.0,
    'Humanities':    3.0,
    'No Major':    4.0,
    'Other':    5.0
}

experience_map = {
    '<1':    0.0,
    '1':    1.0,
    '2':    2.0,
    '3':    3.0,
    '4':    4.0,
    '5':    5.0,
    '6':    6.0,
    '7':    7.0,
    '8':    8.0,
    '9':    9.0,
    '10':    10.0,
    '11':    11.0,
    '12':    12.0,
    '13':    13.0,
    '14':    14.0,
    '15':    15.0,
    '16':    16.0,
    '17':    17.0,
    '18':    18.0,
    '19':    19.0,
    '20':    20.0,
    '>20':    21.0
}

company_type_map = {
    'Pvt Ltd':    0.0,
    'Funded Startup':    1.0,
    'Early Stage Startup':    2.0,
    'Other':    3.0,
    'Public Sector':    4.0,
    'NGO':    5.0
}

company_size_map = {
    '<10':    0.0,
    '10/49':    1.0,
    '100-500':    2.0,
    '1000-4999':    3.0,
    '10000+':    4.0,
    '50-99':    5.0,
    '500-999':    6.0,
    '5000-9999':    7.0
}

last_new_job_map = {
    'never':    0.0,
    '1':    1.0,
    '2':    2.0,
    '3':    3.0,
    '4':    4.0,
    '>4':    5.0
}


def encode_data(json: Dict) -> Dict:
    json['gender'] = gender_map[json['gender']]
    json['enrolled_university'] = enrolled_university_map[json['enrolled_university']]
    json['education_level'] = education_level_map[json['education_level']]
    json['major_discipline'] = major_map[json['major_discipline']]
    json['experience'] = experience_map[json['experience']]
    json['company_size'] = company_size_map[json['company_size']]
    json['company_type'] = company_type_map[json['company_type']]
    json['last_new_job'] = last_new_job_map[json['last_new_job']]
    json['relevent_experience'] = relevent_experience_map[json['relevent_experience']]
    json['training_hours'] = int(json['training_hours'])

    return json
