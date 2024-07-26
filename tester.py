
def reward(soc, threshold, r_max, r_50):
    if soc >= 50:
        return r_max * (soc/50 - 1) + r_50
    elif soc>=threshold:
        return 
    else:
        return 3
    

print(reward(70, 34))