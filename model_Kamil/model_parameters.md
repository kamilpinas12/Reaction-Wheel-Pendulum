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

## Position 0 ( without any mass attached )
a: 0.113, b: -27.233, c: -0.011542 \
Ip: 0.0199, f: -0.0022, ml: 0.0553

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





