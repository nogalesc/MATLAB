function LabelGraph(Student_Name, Main_Title, x_axis, y_axis) 
%This function will set the Title 
%and horizontal and vertical axis labels 
%to match the inputs provided by the user
Current_Date=date;
Main_Title_2=[Student_Name,'    ',Main_Title,'    ',Current_Date];
title(Main_Title_2); 
xlabel(x_axis); 
ylabel(y_axis); 
grid on; 
