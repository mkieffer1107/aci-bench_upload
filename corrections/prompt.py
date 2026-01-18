user_prompt = """\
Your job is to analyze transcriptions of doctor-patient dialogues and find errors where speaker tags are swapped. Each transcript is a list of enumerated lines with speaker tags.

For example, here are some excerpts from a few transcripts where the patient and doctor dialogues are clearly swapped:

Example 1:
23: [patient] alright thanks for coming in today i see on my chart here that you had a bunch of lower respiratory infections so first tell me how are you what's going on\n
24: [doctor] you know i'm doing better now but you know last week i was really sick. I just have had enough like i was coughing a lot a lot of mucus even had some shortness of breath and even a low-grade fever\n
25: [patient] wow that is a lot so what did you do for some of those symptoms\n
26: [doctor] you know i ended up drinking a lot of fluid and taking some robitussin. I actually got better over the weekend and now i'm feeling much better but what concerns me is that i i tend to get pneumonia a lot\n...

Example 2:
8: [patient] hey bruce so see here my my notes here is you here he had positive lab work for hep c so how're you doing today\n
9: [doctor] i'm doing okay but i'm a little bit anxious about having hep c i've really surprised because i've been feeling fine they had done it as you know a screen as just part of my physical so i'm really surprised that that came back positive\n
10: [patient] okay so in the past have any doctors ever told you that you had hep c\n
11: [doctor] no never that's why i'm i'm so surprised\n
12: [patient] okay so just you know i need to ask do you have a history of iv drug use or you know have known any hep c partners\n
13: [doctor] i mean i used to party a lot and even did use iv drugs but i have been clean for over fifteen years now\n

Example 3:
45:[patient] hi good afternoon joseph how are you doing today\n
46:[doctor] i'm doing well but my my big toe hurts and it's a little red too but it really hurts okay how long has this been going on i would say you know off and on for about two weeks but last week is is when it really became painful i was at a a trade show convention and i could n't walk the halls i could n't do anything i just had to stand there and it really hurt the whole time i was there\n
47:[patient] okay does it throb ache burn what kind of pain do you get with it\n
48:[doctor] it's almost like a throbbing pain but occasionally it becomes almost like a a sharp stabbing pain especially if i move it or spend too much time walking i i find myself walking on my heel just to keep that toe from bending\n
49:[patient] okay sorry i got a text and\n
50:[doctor] well that's okay you know what i i you know i what i really you know i love to ride bikes have you you ride bike at all\n

Here is a full transcript for you to analyze:
{transcript}

Please analyze the transcript and find any instances of swapped speaker tags in the patient and doctor dialogues.

Return errors as a list of objects with "lines" and "reason" fields.
- The "lines" field should be a range [start, end] inclusive (e.g., [5, 10] means lines 5 through 10).
- If the entire transcript is swapped, meaning all patient lines are doctor lines and vice versa, use [1, -1] where -1 represents the last line.
- If there are no errors, return a single error with lines [-1, -1] and reason explaining the transcript is correct.
"""

