import face_recognition

def get_first_known_person_name(frame, trained_images_data):
  face_encodings = face_recognition.face_encodings(frame,
                                                   face_recognition.face_locations(frame))
  if len(face_encodings) == 0:
    return "none"
  for face_encoding in face_encodings:
    person_name = get_person_name(face_encoding, trained_images_data)
    if person_name != "unknown":
      return person_name
  return "unknowns"

def get_person_name(face_encoding, trained_images_data):
  counts = get_matched_name_counts(face_encoding, trained_images_data)
  if not counts: # counts is empty
    return "unknown"
  print(counts[max(counts, key=counts.get)])
  return max(counts, key=counts.get)

def get_matched_name_counts(face_encoding, trained_images_data):
  is_matched_trained_images = face_recognition.compare_faces(trained_images_data["encodings"],
                                                             face_encoding)
  counts = {}
  for (idx, is_matched) in enumerate(is_matched_trained_images):
    if is_matched == False:
      continue
    trained_image_person_name = trained_images_data["names"][idx]
    counts[trained_image_person_name] = counts.get(trained_image_person_name, 0) + 1
  return counts
