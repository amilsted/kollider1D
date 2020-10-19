import overlaps as ol

interval = 30
t_max = 568

ol.scan_states_1p("./", interval, t_max, num_ex=64)
