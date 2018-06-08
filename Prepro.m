jpegFiles1 = dir('C:\Users\Amita\Desktop\Amita\Retinal Classification\Images\Training\Diseased\*.jpg'); 
numfiles1 = length(jpegFiles1);
jpegFiles2 = dir('C:\Users\Amita\Desktop\Amita\Retinal Classification\Images\Training\Healthy\*.jpg'); 
numfiles2 = length(jpegFiles2);
tnumfiles=numfiles1+numfiles2;
mydata = cell(1, tnumfiles);
textpart = cell(1,tnumfiles);
PSN= cell(1,tnumfiles);
condition= cell(1,tnumfiles);
isize= cell(1,tnumfiles);

str1='C:/Users/Amita/Desktop/Amita/Retinal Classification/Images/Training/Diseased/%s'; 
for k = 1:numfiles1
    filename=jpegFiles1(k).name;
    str2=sprintf(str1,filename);
    mydata{k} = imread(str2);
    mydata{k} = rgb2gray(mydata{k});
    textpart{k}= imcrop(mydata{k},[110,0,350,85]);
    ocrtext=ocr(textpart{k});
    PSN{k}=strtrim(ocrtext.Text);
    condition{k}=1;
    isize{k}=size(mydata{k});
end

str3='C:/Users/Amita/Desktop/Amita/Retinal Classification/Images/Training/Healthy/%s'; 
for k = numfiles1+1:tnumfiles
    m=k-numfiles1;
    filename=jpegFiles2(m).name;
    str4=sprintf(str3,filename);
    mydata{k} = imread(str4);
    mydata{k} = rgb2gray(mydata{k});
    textpart{k}= imcrop(mydata{k},[110,0,350,85]);
    ocrtext=ocr(textpart{k});
    PSN{k}=strtrim(ocrtext.Text);
    condition{k}=0;
    isize{k}=size(mydata{k});
end
I=mydata{10};
npixels=size(I,1)*size(I,2);
R=reshape(I,[1,npixels]);
R(R<15)=[];
boxplot(R);
PSN=[{'PSN'},PSN];
mydata=[{'Images'},mydata];
condition=[{'Condition'},condition];
isize=[{'Image size'},isize];
data=[PSN',mydata',isize',condition'];
idataset=cell2dataset(data);


