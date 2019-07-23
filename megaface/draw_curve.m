% requirement: JSONLab: https://cn.mathworks.com/matlabcentral/fileexchange/33381-jsonlab--a-toolbox-to-encode-decode-json-files

format long
addpath('D:\Users\foamliu\code\jsonlab');
facescrub_cmc_file = 'D:\Users\foamliu\code\InsightFace-v3\megaface\results\cmc_facescrub_megaface_0_1000000_1.json'
facescrub_cmc_json = loadjson(fileread(facescrub_cmc_file));
facescrub_cmc_json


figure(1);
semilogx(facescrub_cmc_json.cmc(1,:)+1,facescrub_cmc_json.cmc(2,:)*100,'LineWidth',2);
title(['Identification @ 1e6 distractors = ' num2str(facescrub_cmc_json.cmc(2,:)(1))]);
xlabel('Rank');
ylabel('Identification Rate %');
ylim([0 100]);
grid on;
box on;
hold on;

facescrub_cmc_json.roc(1,:)

figure(2);
%semilogx(facescrub_cmc_json.roc(1,:),facescrub_cmc_json.roc(2,:),'LineWidth',2);
xdata=[0.0,			1.028475438147325e-08,			3.085426314441975e-08,			2.674036068128771e-07,			1.069614427251508e-06,			8.011823410924990e-06,			0.0003234863688703626,			1.0		],
ydata=[0.9215109944343567,			0.9381747841835022,			0.9535316228866577,			0.9636464118957520,			0.9736617803573608,			0.9836629033088684,			0.9936640262603760,			1.0]
semilogx(xdata,ydata,'LineWidth',2);
%semilogx(facescrub_cmc_json.roc{1},facescrub_cmc_json.roc{2},'LineWidth',2);
title(['Verification @ 1e-6 = ' num2str(interp1(xdata, ydata, 1e-6))]);
xlim([1e-6 1]);
%ylim([0 1]);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
grid on;
box on;
hold on;