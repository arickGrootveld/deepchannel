% Co-ordinated turn model
function [f2, F_transition] = c_turn(x,param)
    dt = param;
    if x(5) == 0
        f2 =     [x(1) + x(3) * dt;
                  x(2) + x(4) * dt;
                  x(3);
                  x(4);
                  0              ];
         F_transition = [1, 0, dt, 0;
                         0, 1, 0, dt;
                         0, 0, 1, 0;
                         0, 0, 0, 1];
         test = F_transition * x(1:4);
    else
        wt = x(5) * dt;
        w = x(5);
        f2 =     [x(1) + x(3)/w * sin(wt) - x(4)/w * (1 - cos(wt));
                  x(2) + x(3)/w* (1 - cos(wt)) + x(4)/w*sin(wt);
                  x(3)*cos(wt) - x(4) * sin(wt);
                  x(3) * sin(wt) + x(4)* cos(wt);
                  w];
        F_transition = [1, 0, 1/w * sin(wt), -1/w * (1 - cos(wt));
                        0, 1, 1/w * (1-cos(wt)), 1/w*sin(wt);
                        0, 0, cos(wt), -1*sin(wt);
                        0, 0, sin(wt), cos(wt)];
        test = F_transition * x(1:4);
    end

end 