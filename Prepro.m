% SETTING UP ALL THE CELLS
jpegFiles1 = dir('C:\Users\Amita\Desktop\Amita\Retinal Classification\Images\diaretdb1_v_1_1\resources\images\ddb1_fundusimages\Diseased\*.png'); 
numfiles1 = length(jpegFiles1);
jpegFiles2 = dir('C:\Users\Amita\Desktop\Amita\Retinal Classification\Images\diaretdb1_v_1_1\resources\images\ddb1_fundusimages\Healthy\*.png'); 
numfiles2 = length(jpegFiles2);
tnumfiles=numfiles1+numfiles2;
mydata = cell(1, tnumfiles);
rgbim=cell(1,tnumfiles);
textpart = cell(1,tnumfiles);
PSN= cell(1,tnumfiles);
condition= cell(1,tnumfiles);
isize= cell(1,tnumfiles);


% READ DISEASED IMAGES INTO CELLS
str1='C:/Users/Amita/Desktop/Amita/Retinal Classification/Images/diaretdb1_v_1_1/resources/images/ddb1_fundusimages/Diseased/%s';  
for k = 1:numfiles1
    filename=jpegFiles1(k).name;
    str2=sprintf(str1,filename);
    rgbim{k} = imread(str2);
    mydata{k} = rgb2gray(rgbim{k});
    % mydata{k}=imcrop(mydata{k},[0 38 2448 3188]);
    textpart{k}= imcrop(mydata{k},[110,0,350,85]);
    ocrtext=ocr(textpart{k});
    PSN{k}=strtrim(ocrtext.Text);
    condition{k}=1;
    isize{k}=size(mydata{k});
end

% READ HEALTHY IMAGES INTO CELLS
str3='C:/Users/Amita/Desktop/Amita/Retinal Classification/Images/diaretdb1_v_1_1/resources/images/ddb1_fundusimages/Healthy/%s';
for k = numfiles1+1:tnumfiles
    m=k-numfiles1;
    filename=jpegFiles2(m).name;
    str4=sprintf(str3,filename);
    rgbim{k} = imread(str4);
    mydata{k} = rgb2gray(rgbim{k});
    % mydata{k}=imcrop(mydata{k},[0 38 2448 3188]);
    textpart{k}= imcrop(mydata{k},[110,0,350,85]);
    ocrtext=ocr(textpart{k});
    PSN{k}=strtrim(ocrtext.Text);
    condition{k}=0;
    isize{k}=size(mydata{k});
end

% MAKE BOXPLOT
% I=mydata{10};
% npixels=size(I,1)*size(I,2);
% R=reshape(I,[1,npixels]);
% R(R<15)=[];
% figure(1);boxplot(R);

% FORM DATA FRAME
PSN=[{'PSN'},PSN];
mydata=[{'Images'},mydata];
condition=[{'Condition'},condition];
isize=[{'Image size'},isize];
data=[PSN',mydata',isize',condition'];
idataset=cell2dataset(data);

% STEP BY STEP FOR MORPHOLOGICAL PROCESSING
% I=imread('C:/Users/Amita/Desktop/Amita/Retinal Classification/Images/diaretdb1_v_1_1/resources/images/ddb1_fundusimages/image028.png');
% I=rgb2gray(I);
figure(2);subplot(2,2,1);imshow(I)
title('Original')
Icontrast=imadjust(I);
subplot(2,2,2);imshow(Icontrast)
title('Increased contrast')
Ibinary=imbinarize(Icontrast,'adaptive');
subplot(2,2,3);imshow(Ibinary)
title('Binary')
Iopen=bwareaopen(Ibinary,100);
subplot(2,2,4);imshow(Iopen)
title('Morphological processing')

% MORPHOLOGICAL PROCESSING
for i=1:tnumfiles
    I=mydata{i};
    Icontrast=imadjust(I);
    Ibinary=imbinarize(Icontrast,'adaptive');
    Iopen=bwareaopen(Ibinary,100);
end

% FILTERING 
for i=1:tnumfiles
    I=mydata{i};
    Icontrast=adapthisteq(I);
    I2=imgaussfilt(Icontrast,256);
    I3=imabsdiff(I,I2);
    I4=imabsdiff(I,I3);
end
