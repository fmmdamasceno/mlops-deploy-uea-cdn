from typing import Dict

from models import RequestBodyModel


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

city_indexes_map = {
    "0": 0.847,
    "1": 0.895,
    "2": 0.887,
    "3": 0.558,
    "4": 0.804,
    "5": 0.92,
    "6": 0.924,
    "7": 0.794,
    "8": 0.698,
    "9": 0.518,
    "10": 0.701,
    "11": 0.55,
    "12": 0.698,
    "13": 0.926,
    "14": 0.789,
    "15": 0.743,
    "16": 0.698,
    "17": 0.722,
    "18": 0.64,
    "19": 0.78,
    "20": 0.781,
    "21": 0.738,
    "22": 0.479,
    "23": 0.745,
    "24": 0.527,
    "25": 0.625,
    "26": 0.827,
    "27": 0.68,
    "28": 0.742,
    "29": 0.698,
    "30": 0.897,
    "31": 0.836,
    "32": 0.487,
    "33": 0.698,
    "34": 0.856,
    "35": 0.763,
    "36": 0.727,
    "37": 0.74,
    "38": 0.84,
    "39": 0.555,
    "40": 0.735,
    "41": 0.689,
    "42": 0.698,
    "43": 0.698,
    "44": 0.556,
    "45": 0.769,
    "46": 0.766,
    "47": 0.843,
    "48": 0.91,
    "49": 0.92,
    "50": 0.767,
    "51": 0.903,
    "52": 0.649,
    "53": 0.921,
    "54": 0.664,
    "55": 0.878,
    "56": 0.776,
    "57": 0.764,
    "58": 0.512,
    "59": 0.824,
    "60": 0.698,
    "61": 0.682,
    "62": 0.788,
    "63": 0.796,
    "64": 0.624,
    "65": 0.899,
    "66": 0.698,
    "67": 0.698,
    "68": 0.698,
    "69": 0.848,
    "70": 0.939,
    "71": 0.698,
    "72": 0.807,
    "73": 0.448,
    "74": 0.893,
    "75": 0.794,
    "76": 0.898,
    "77": 0.776,
    "78": 0.827,
    "79": 0.563,
    "80": 0.516,
    "81": 0.725,
    "82": 0.89,
    "83": 0.762,
    "84": 0.493,
    "85": 0.896,
    "86": 0.74,
    "87": 0.856,
    "88": 0.739,
    "89": 0.866,
    "90": 0.775,
    "91": 0.913,
    "92": 0.645,
    "93": 0.666,
    "94": 0.802,
    "95": 0.855,
    "96": 0.856,
    "97": 0.647,
    "98": 0.698,
    "99": 0.884,
    "100": 0.795,
    "101": 0.754,
    "102": 0.579,
    "103": 0.939,
    "104": 0.698,
    "105": 0.83,
    "106": 0.579,
    "107": 0.698,
    "108": 0.698,
    "109": 0.847,
    "110": 0.73,
    "111": 0.693,
    "112": 0.923,
    "113": 0.698,
    "114": 0.925,
    "115": 0.743,
    "116": 0.698,
    "117": 0.691,
    "118": 0.865,
    "119": 0.698,
    "120": 0.925,
    "121": 0.949,
    "122": 0.915
}


def map_data(body: RequestBodyModel) -> Dict:
    json = {}
    json['gender'] = gender_map[body.gender]
    json['enrolled_university'] = enrolled_university_map[body.enrolled_university]
    json['education_level'] = education_level_map[body.education_level]
    json['major_discipline'] = major_map[body.major_discipline]
    json['experience'] = experience_map[body.experience]
    json['company_size'] = company_size_map[body.company_size]
    json['company_type'] = company_type_map[body.company_type]
    json['last_new_job'] = last_new_job_map[body.last_new_job]
    json['city'] = int(body.city)
    json['city_development_index'] = city_indexes_map[body.city]
    json['relevent_experience'] = relevent_experience_map[body.relevent_experience]
    json['training_hours'] = int(body.training_hours)

    return json
