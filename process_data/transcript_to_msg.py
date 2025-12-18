# Note: No timestamps being saved

# GLOBAL
PATIENT_NAMES = ["PATIENT"]
THERAPIST_NAMES = ["THERAPIST"]

# UTILITIES
def starts_with_selection(target, startswith_list, case_sensitive=True):
    if not case_sensitive:
        # Make copies
        target = target.copy()
        startswith_list = startswith_list.deepcopy()

        # Convert to Lower
        target = target.lower()
        startswith_list = [x.lower() for x in startswith_list]

    for startswith in startswith_list:
        if target.startswith(startswith):
            return True
    return False

def strip_list(target, s_list, space_strip=True):
    for strip in s_list:
        target = target.strip(strip)

    if space_strip:
        target = target.strip()

    return target

def load_and_split(filename):
    # Load File
    with open(filename) as f:
        lines = f.readlines()
    # print((lines))

    # Process into a clean list
    start_file = False
    r_list = []
    messages = []
    for line in lines:
        line = strip_list(line, ["</p>", "<p>", "</p>"])

        if starts_with_selection(line, PATIENT_NAMES + THERAPIST_NAMES): # Add counseler
            start_file = True
        if not start_file or len(line) == 0:
            continue

        if starts_with_selection(line, THERAPIST_NAMES):
            content = strip_list(line, THERAPIST_NAMES)
            content = content.strip(": ")
            entry = {"role": "assistant", "content": content}
            messages.append(entry)

        if starts_with_selection(line, PATIENT_NAMES):
            content = strip_list(line, PATIENT_NAMES)
            content = content.strip(": ")
            entry = {"role": "user", "content": content}
            messages.append(entry)


        r_list.append(line)

    return messages



if __name__ == '__main__':
    load_and_split("/Users/danielarturi/Desktop/PersonalProjects/AlgoverseFall2/data/test/6_1000088056.txt")