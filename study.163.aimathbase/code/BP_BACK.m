%-------------����˵��----------------  
%    �����ص�˳��  
%       ���������  
%               training_example:��ѵ��������  
%                    eta�� ѧϰ����  
%---------------------------------------  
function BP_BACK(training_example,eta)  
[m,n] = size(training_example); %m--�У�n--��  
%��ʼ��Ȩֵ����-0.5~0.5֮��  
w = rand(2,3) - 0.5;      
v = rand(3,2) - 0.5;  
u = rand(2,3) - 0.5;  
%------------------------  
for num = 1:n  %��ÿ���������������  
%%----------------------------------------  
        one_sample = training_example(:,num); %ȡһ��  
        x = one_sample(1:3);        %��ȡ���е�����  
        y = one_sample(4:5);        %��ȡ���е����  
        net2 = w * x;               %��һ�����ֵ  
        for i=1:2  
             hidden1(i)=1/(1+exp(-net2(i)));%����sigmoid�������  
        end  
        net3 = v * hidden1';      %�ڶ������ֵ   
        for i=1:3  
             hidden2(i)=1/(1+exp(-net3(i)));%����sigmoid�������  
        end  
        net4 = u * hidden2';      %���������ֵ  
        for i=1:2  
            o(i)=1/(1+exp(-net4(i)));%sigmoid������������յ����ֵo  
        end  
%%-------------���򴫲��㷨���������deltaֵ�����E�Ը���Ȩֵ�ĵ�����-----------------  
       %���һ��deltaֵ  
        for i = 1:2  
             delta3(i) = (y(i)-o(i))*o(i)*(1-o(i)); %���㹫ʽ  
        end  
       %-----�ڶ���������---  
        for j = 1:3      %���㹫ʽ�������һ���deltaֵ���  
             delta2(j) = hidden2(j)*(1-hidden2(j))*delta3*u(:,j);  
        end  
       %-----��һ��������---  
        for k = 1:2      %���㹫ʽ�������һ���deltaֵ���  
             delta1(k) = hidden1(k)*(1-hidden1(k))*delta2*v(:,k);  
        end  
%---------------------------------------------------------  
%--------����delta�������ʼ����Ȩֵ---------------------  
%---����uȨֵ-----  
        for i = 1:2   
               for j = 1:3      %���㹫ʽ  w = w + eta*delta*x  
                     u(i,j) = u(i,j) + eta*delta3(i)*hidden2(j);  
               end  
        end  
%---����vȨֵ-----  
        for i = 1:3  
                for j = 1:2  
                     v(i,j) = v(i,j) + eta*delta2(i)*hidden1(j);  
                end  
        end  
%---����wȨֵ-----  
        for i = 1:2  
                for j = 1:3  
                     w(i,j) = w(i,j) + eta*delta1(i)*x(j);  
                end  
        end  
%---------------------------------------------------------  
%--------------��¼һ��������̺�����ֵ  
            e=o'-y;%�����������  ���������-Ŀ�������  
            sigma(num)=e'*e;%�������ƽ����  
end  
%%-------------------------------------------------------  
plot(sigma); 