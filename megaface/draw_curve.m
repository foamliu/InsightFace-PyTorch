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
%ylim([0 100]);
grid on;
box on;
hold on;

facescrub_cmc_json.roc(1,:)

figure(2);
%semilogx(facescrub_cmc_json.roc(1,:),facescrub_cmc_json.roc(2,:),'LineWidth',2);
xdata=[0.0,			6.170852628883949e-08,			1.851255859719458e-07,			7.816413472028216e-07,			4.607570190273691e-06,			5.529083864530548e-05,			1.0		],
ydata=[0.9389561414718628,			0.9498948454856873,			0.9601233005523682,			0.9702096581459045,			0.9802108407020569,			0.9902119636535645,			1.0]
semilogx(xdata,ydata,'LineWidth',2);
%semilogx(facescrub_cmc_json.roc{1},facescrub_cmc_json.roc{2},'LineWidth',2);
title(['Verification @ 1e-6 = ' num2str(interp1(xdata, ydata, 1e-6))]);
xlim([1e-6 1]);
ylim([0 1]);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
grid on;
box on;
hold on;