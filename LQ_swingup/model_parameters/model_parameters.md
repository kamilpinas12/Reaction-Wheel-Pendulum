# Model with non physical variables
$ \dot{x}_1 = x_2 $ \
$ \dot{x}_2 = -ax_2 + bsin(x_1) + cK(u - dx_3) $ \
$ \dot{x}_3 = K(u - dx_3) $

# Model with physical variables

$ \dot{x}_1 = x_2 $ \
$ \dot{x}_2 = +\frac{f}{J_p}x_2 + \frac{mgl}{J_p}sin(x_1) +\frac{J_w}{J_p} K(u - dx_3) $ \
$ \dot{x}_3 = K(u - dx_3) $

(signs can be messed up XD i needed to change signs because of negative mass length and friction ...)

### To convert between above representation you can use: `convert_parameters.py`


## Motor params ( const )
Iw = 0.00023
Km = 484.73;
d = 0.00229;

## parametry z ciężarkiem w pozycji domyślnej ( tej z pierwszych zajęć (na kresce) )
a: 0.11655, b: -3.91501, c: -0.00810
Ip: 0.02838, f: 0.00331, ml: 0.01133

## Position 0 ( without any mass attached )
a: 0.113, b: -27.233, c: -0.011542 \
Ip: 0.0199, f: 0.0022, ml: 0.0553

## Position 1 
a: 0.085449, b: -11.498262, c: -0.009128 \
Ip: 0.0251, f: 0.00215, ml: 0.02953

## Position 2
a: 0.085634, b: -9.101332, c: -0.009168 \
Ip: 0.02508, f: 0.00214, ml: 0.02327

## Position 3 
a: 0.08683, b: -6.959842, c: -0.008187 \
Ip: 0.028093, f: 0.002439, ml: 0.019931


### Ip and ml are corelated with equation 
$ ml = -4.52 * I_p + 0.14 $\
$ I_p = -0.221* ml + 0.031 $ 


## ident_square_53.mat
a: 0.05416, b: -1.93161, c: -0.00802
Ip: 0.02866, f: 0.00155, ml: 0.00564

## ident_square_85.mat
a: 0.29969, b: -4.01598, c: -0.00786
Ip: 0.02927, f: 0.00877, ml: 0.01198

## ident_square_106.mat
a: 0.10484, b: -5.73889, c: -0.00895
Ip: 0.02568, f: 0.00269, ml: 0.01503

## ident_square_135.mat
a: 0.07803, b: -8.38395, c: -0.00972
Ip: 0.02365, f: 0.00185, ml: 0.02021




