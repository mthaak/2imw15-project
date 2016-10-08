import pickle

users = ["twitter"]

pickle.dump(users, open("users.p", "wb"))
