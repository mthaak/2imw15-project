import pickle


objects = []
with (open("../Data/features.csv", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

for w in object:
    print(w)