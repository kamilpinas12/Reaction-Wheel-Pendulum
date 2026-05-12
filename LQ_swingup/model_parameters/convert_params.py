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
    fmt = lambda x: f"{x:.5f}"

    a = 0.054164
    b = -1.931606
    c = -0.008024
    Ip, f, ml = convert_to_physical(a, b, c)
    print(f"a: {fmt(a)}, b: {fmt(b)}, c: {fmt(c)}")
    print(f"Ip: {fmt(Ip)}, f: {fmt(f)}, ml: {fmt(ml)}")
    print("=============================")


    a = 0.299692
    b = -4.015983
    c = -0.007858
    Ip, f, ml = convert_to_physical(a, b, c)
    print(f"a: {fmt(a)}, b: {fmt(b)}, c: {fmt(c)}")
    print(f"Ip: {fmt(Ip)}, f: {fmt(f)}, ml: {fmt(ml)}")
    print("=============================")

    a = 0.078027
    b = -8.383954
    c = -0.009725
    Ip, f, ml = convert_to_physical(a, b, c)
    print(f"a: {fmt(a)}, b: {fmt(b)}, c: {fmt(c)}")
    print(f"Ip: {fmt(Ip)}, f: {fmt(f)}, ml: {fmt(ml)}")
    print("=============================")

    a = 0.104845
    b = -5.738891
    c = -0.008955
    Ip, f, ml = convert_to_physical(a, b, c)
    print(f"a: {fmt(a)}, b: {fmt(b)}, c: {fmt(c)}")
    print(f"Ip: {fmt(Ip)}, f: {fmt(f)}, ml: {fmt(ml)}")
    print("=============================")

    # check correctness of conversion
    # Ip = 0.028093318675949677 
    # f = 0.002439342860632710
    # ml = 0.019931198699312837

    # a, b, c = convert_to_non_physical(Ip, f, ml)
    # print(f"Ip: {Ip}, f: {f}, ml: {ml}")
    # print(f"a: {a}, b: {b}, c: {c}")