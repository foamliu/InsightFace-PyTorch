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
xdata=[0.0,			1.034335816996190e-08,			4.137343267984761e-08,			5.171678907345267e-08,			7.240350896609016e-08,			1.137769416459378e-07,			1.965237999002056e-07,			1.841117750700505e-06,			0.001009718631394207,			1.0		],
ydata=[			0.8872767686843872,			0.9050645828247070,			0.92467862367630,			0.9406545758247375,			0.9531793594360352,			0.9637916684150696,			0.9753674268722534,			0.9853757619857788,			0.9953840970993042,			1.0]
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