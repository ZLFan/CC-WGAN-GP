for i=1:757
    path=strcat(int2str(i-1),'.mat')
    loadpath=strcat('erp/',int2str(i-1),'.jpg')
    load(path);
    a=squeeze(eeg)
    a=mean(a,1);
    fig = figure; % �½�һ��figure������ͼ�������浽fig
    plot(a) % ��"."����ʽ��x��y�������������ɵ�ͼ����
    frame = getframe(fig) % ��ȡframe
    img = frame2im(frame); % ��frame�任��imwrite��������ʶ��ĸ�ʽ
    imwrite(img,loadpath); % ���浽����Ŀ¼�£�����Ϊ"a.png"
    close all
end
 erp