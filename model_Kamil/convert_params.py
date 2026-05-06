def convert_to_physical(a, b, c):
    #Constant params:
    Iw = 0.00023
    Km = 484.73
    d = 0.00229
    g = 9.81

    Ip = -Iw/c
    f = a*Ip
    ml = -b*Ip/g

    return Ip, f, ml


def convert_to_non_physical(Ip, f, ml):
    #Constant params:
    Iw = 0.00023
    Km = 484.73
    d = 0.00229
    g = 9.81
    
    c = -Iw/Ip
    a = f/Ip
    b = -(ml*g)/Ip

    return a, b, c



if __name__ == "__main__":
    a = 0.113
    b=-27.233
    c = -0.011542
    Ip, f, ml = convert_to_physical(a, b, c)
    print(f"a: {a}, b: {b}, c: {c}")
    print(f"Ip: {Ip}, f: {f}, ml: {ml}")
    print("=============================")


    a = 0.085449
    b = -11.498262
    c = -0.009128
    Ip, f, ml = convert_to_physical(a, b, c)
    print(f"a: {a}, b: {b}, c: {c}")
    print(f"Ip: {Ip}, f: {f}, ml: {ml}")
    print("=============================")

    a = 0.085634 
    b = -9.101332
    c = -0.009168
    Ip, f, ml = convert_to_physical(a, b, c)
    print(f"a: {a}, b: {b}, c: {c}")
    print(f"Ip: {Ip}, f: {f}, ml: {ml}")
    print("=============================")

    a = 0.086830
    b = -6.959842
    c = -0.008187
    Ip, f, ml = convert_to_physical(a, b, c)
    print(f"a: {a}, b: {b}, c: {c}")
    print(f"Ip: {Ip}, f: {f}, ml: {ml}")
    print("=============================")

    # check correctness of conversion
    # Ip = 0.028093318675949677 
    # f = 0.002439342860632710
    # ml = 0.019931198699312837

    # a, b, c = convert_to_non_physical(Ip, f, ml)
    # print(f"Ip: {Ip}, f: {f}, ml: {ml}")
    # print(f"a: {a}, b: {b}, c: {c}")