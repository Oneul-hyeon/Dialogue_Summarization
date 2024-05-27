def preprocess(text) :
    prev_user = -1
    utter = ""
    dialogue = []
    text = text.strip()
    split_text = list(text.split("\n"))

    for utterance in split_text :
        idx = utterance.index(":")
        user = utterance[:idx].rstrip()
        utter_ = utterance[idx+1:].lstrip()
        if prev_user == -1 :
            utter = utter_
            prev_user = user
        elif prev_user == user :
            utter += ' ' + utter_
        else :
            dialogue.append(utter)
            utter = utter_
            prev_user = user

    return dialogue