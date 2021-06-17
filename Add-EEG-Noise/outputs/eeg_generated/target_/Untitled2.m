for i=1:757
    path=strcat(int2str(i-1),'.mat')
    loadpath=strcat('erp/',int2str(i-1),'.jpg')
    load(path);
    a=squeeze(eeg)
    a=mean(a,1);
    fig = figure; % 新建一个figure，并将图像句柄保存到fig
    plot(a) % 用"."的形式将x，y表现在上面生成的图像中
    frame = getframe(fig) % 获取frame
    img = frame2im(frame); % 将frame变换成imwrite函数可以识别的格式
    imwrite(img,loadpath); % 保存到工作目录下，名字为"a.png"
    close all
end
 erp