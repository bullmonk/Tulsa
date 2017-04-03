clear;clc;close all

%--------------------------------------------------------------------------
PlotsDir='./Plots/';
if ~exist(PlotsDir,'dir')
    command=['mkdir -pv ' PlotsDir];
    disp(command);
    system(command);
end

%--------------------------------------------------------------------------
iverb=0;

ispecify=1;

%--------------------------------------------------------------------------
WantedName={...
    'Lag_15_MaxDISPH','Lag_26_MinGRN','MinZ0H','Lag_9_MeanEFLUX',...
    'Lag_9_MaxEVPTRNS','Lag_12_MeanEVPTRNS','Lag_9_MeanEVPTRNS',...
    'Lag_14_MeanEVPTRNS','Lag_11_MeanEVPTRNS','Lag_12_MaxEVPTRNS',...
    'Lag_8_MaxEVPTRNS','Lag_11_MaxEVPTRNS','Lag_14_MaxEVPTRNS',...
    'Lag_30_MinTELAND','MeanTSH','Lag_29_MinTELAND','Lag_1_MeanTSH',...
    'MaxHLML','Lag_8_MaxEFLUX','Lag_29_MaxTELAND','MeanTUNST',...
    'Lag_1_MeanTUNST','Lag_8_MaxEVAP','MeanTWLT'...
    };

% WantedName={...
%     'Lag_15_MaxDISPH','MinZ0H',...
%     'Lag_30_MinTELAND','Lag_29_MinTELAND',...
%     'Lag_29_MaxTELAND'
%     };

%--------------------------------------------------------------------------
% We will be creating some plots, set up the plot size and the screen 
% location for the first plot
width=600;
height=400;
left=1;
bottom=1;

jfigure=0;

Renderer='painters';
Renderer='opengl';

%--------------------------------------------------------------------------
% Find machine specs
[ngpus,ncores,ncpus,archstr,maxsize,endian] = findcapabilitiescomputer();

if ngpus>0
    useGPU='yes';
else
    useGPU='no';
end
if ncpus>1
    useParallel='yes';
else
    useParallel='no';
end

%--------------------------------------------------------------------------
% Loading all the data
disp('Loading all the data')
tic
load('AmbrosiaTulsaTraining.mat')
toc

%--------------------------------------------------------------------------
% Set up the machine learning matrices
disp('Set up the machine learning matrices')
tic
command=['Y=Out;X=[In'];
for i=1:30
    command=[command ' TS_' num2str(i)];
end
command=[command '];'];
disp(command);
eval(command);
toc

%--------------------------------------------------------------------------
% Set up the variable names
disp('Set up the machine learning matrices')
tic
icount=0;
% time lag loop
for i=0:30
    % loop over variables for this time lag
    for j=1:length(namesTS)
        icount=icount+1;
        if i==0
            names{icount}=namesTS{j};
        else
            command=['names{icount}=namesTS_' num2str(i) '{j};'];
            if iverb
                disp(command);
            end
            eval(command);
        end
    end
    % loop over variables for this time lag
end
% time lag loop
toc

whos names

%--------------------------------------------------------------------------
% Drop rows with missing values
disp('remove rows with missing pollen values')
whos X Y
[row,col]=find(isnan(Y));
X(row,:)=[];
Y(row)=[];
whos X Y
N=length(Y);

%--------------------------------------------------------------------------
% Drop columns which are constant
disp('Drop columns which are constant or have NaN')
nx=size(X);
ikeep=[];
for i=1:nx(2)
    clear x iwant
    x=squeeze(X(:,i));
    if abs(range(x))>0 & length(find(isnan(x)))==0
        ikeep=[ikeep i];
    end
end
% just keep rows that have valid non-constant data
clear x
x=X(:,ikeep);
X=x;
clear x

clear keptnames
for i=1:length(ikeep)
    keptnames{i}=names{ikeep(i)};
end
clear names
names=keptnames;
clear keptnames

whos X Y

%--------------------------------------------------------------------------
% Create a table from the X array
Data=array2table(X,'VariableNames',names);
Data.Pollen=Y;

%--------------------------------------------------------------------------
nx=size(X);
if iverb
    for i=1:nx(2)
        clear x iwant
        x=squeeze(X(:,i));
        plot(x)
        title(names{i})
        pause
    end
end

%--------------------------------------------------------------------------
if ispecify
    AllNamesWeights=WantedName;
else
    % use fsrnca to choose the most important features
    fn_all_mat=['ImptFeatures.mat'];
    if ~exist(fn_all_mat,'file')

        for iter=1:10

            fn=['ImptFeatures-' num2str(iter)];
            fn_mat=[fn '.mat'];

            if ~exist(fn_mat,'file')

                %--------------------------------------------------------------------------
                % Fit the NCA model.
                nca = fsrnca(...
                    X,Y,...
                    'FitMethod','average',...
                    'Solver','minibatch-lbfgs',...
                    'Standardize',1,...
                    'Verbose',1,...
                    'Lambda',0.5/N...
                    );    

                %--------------------------------------------------------------------------
                % Feature weights
                FW=mean(nca.FeatureWeights,2);
                [FWs,ifw] = sort(FW,'descend');

                %--------------------------------------------------------------------------
                % Find most common Weight
                [Ncounts,EDGES] = histcounts(log10(FWs),100);
                [max_counts,imax_counts] = max(Ncounts);

                % find just the weights that are greater than zero
                iwant=find(log10(FWs)>=log10(FWs(imax_counts)));

                % Labels
                for i=1:length(iwant)
                    NamesWeights{i}=names{ifw(i)};
                end


                figure
                plot_FWS=log10(FWs);
                barh(plot_FWS(iwant));
                yt = get(gca,'YTick');    
                set(gca,'TickDir','out');
                set(gca, 'ydir', 'reverse' )

                % Add text labels to each bar
                for ii=1:length(iwant)
                    text(...
                        double(max([0 plot_FWS(ii)+0.01*max(plot_FWS)])),ii,...
                        NamesWeights{ii},'Interpreter','latex','FontSize',11);
                end
                xl=xlim;
                xlim([xl(1) 1.5*xl(2)])

                set(gca,'LineWidth',2);   

                title(num2str(iter))
                drawnow

                [fnpng]=wrplotpng([PlotsDir fn]);

                command=['save ' fn_mat ' NamesWeights'];
                disp(command);
                eval(command);

            else

                disp(fn_mat)
                load(fn_mat)

            end


            if iter==1
                AllNamesWeights=NamesWeights;whos AllNamesWeights
            else
                AllNamesWeights=unique([AllNamesWeights NamesWeights]);whos AllNamesWeights
            end

        end

        command=['save ' fn_all_mat ' AllNamesWeights'];
        disp(command);
        eval(command);

    else
       disp(fn_all_mat)
       load(fn_all_mat)
    end
    
end

disp(['# of important features is: ' num2str(length(AllNamesWeights))])
for i=1:length(AllNamesWeights)
    disp(AllNamesWeights{i})
end
disp(['# of important features is: ' num2str(length(AllNamesWeights))])

%--------------------------------------------------------------------------
% Set up the data for the machine learning

%--------------------------------------------------------------------------
% Set up the X & Y arrays
command=['InAll=double(['];
for i=1:length(AllNamesWeights)
    if length(AllNamesWeights{i})>0
        command=[command 'Data.' AllNamesWeights{i}];
        if i<length(AllNamesWeights)
            command=[command ' '];
        end
    end
end
command=[command ']);'];
disp(command)
eval(command);

%--------------------------------------------------------------------------
YNames={'Pollen'};
command=['OutAll=['];
for i=1:length(YNames)
    command=[command 'Data.' YNames{i}];
    if i<length(YNames)
        command=[command ' '];
    end
end
command=[command '];'];
disp(command)
eval(command);

%--------------------------------------------------------------------------
% Split up the data to provide a training dataset and independent validation 
% dataset of size specified by the validation fraction, validation_fraction
validation_fraction=0.1;

disp(['Validation Fraction is: ' num2str(validation_fraction)])
ipointer=1:length(OutAll);
cvpart=cvpartition(ipointer,'holdout',validation_fraction);

% Training Data
InTrain=InAll(training(cvpart),:);
OutTrain=OutAll(training(cvpart),:);

% Independent Validation Data
InTest=InAll(test(cvpart),:);
OutTest=OutAll(test(cvpart),:);

%--------------------------------------------------------------------------
% fit counter
ifit=0;

% correlation coefficent threshold for listing good fits
threshold_r=0.5;

% cross validation switch (0=off, 1=on)
icrossval=1;

%--------------------------------------------------------------------------
% Set up the random forest parameters
ntrees=80;
leaf = 1;
niter=10;
fboot=1;

% Set up the size of the parallel pool
npool=ncores;

% Opening parallel pool
if ncpus>1
    tic
    disp('Opening parallel pool')

    % first check if there is a current pool
    poolobj=gcp('nocreate');

    % If there is no pool create one
    if isempty(poolobj)
        command=['parpool(' num2str(npool) ');'];
        disp(command);
        eval(command);
    else
        poolsize=poolobj.NumWorkers;
        disp(['A pool of ' poolsize ' workers already exists.'])
    end

    % Set parallel options
    paroptions = statset('UseParallel',true);
    toc

end


clear mdlRF

% start iteration loop    
for iter=1:niter

    if iter==1
        ntrees=50;
    else
        ntrees=14;
    end
        
    
    description=['Random Forest with ' num2str(ntrees) ' trees'];
    disp(description)    
    b=TreeBagger(...
        ntrees,InTrain,OutTrain,...
        'Method','regression',...
        'oobvarimp','on',...
        'minleaf',leaf,...
        'FBoot',fboot,...
        'Options',paroptions...        
        );
    
    mdlRF{iter}=b;

    fit=predict(b,InTest);
    fit_t=predict(b,InTrain);
    fit_err_t=InTrain-fit_t;
    
    this_r=isnancorr(OutTest,fit);
    this_t=isnancorr(OutTrain,fit_t);

    r_iter_t(iter)=this_t;
    r_iter_v(iter)=this_r;    
    
    disp(['RF ' num2str(iter) ' has R value of ' num2str(this_r)])    

    %--------------------------------------------------------------------------
    % Out of bag error over the number of grown trees
    oobErr=oobError(b);

    %--------------------------------------------------------------------------
    % Ploting how weights change with variable rank
    disp('Ploting out of bag error over the number of grown trees')

    %--------------------------------------------------------------------------
    figure('Position',[left, bottom, width, height],'Renderer',Renderer)

    %--------------------------------------------------------------------------
    jfigure=jfigure+1;%figure positioning
    left=left+width;
    if left>4*width
        left=1;
        %bottom=bottom+height;
        if bottom>4*height
            bottom=1;
        end
    end

    %--------------------------------------------------------------------------
    %plot out of bag error
    plot(oobErr,'LineWidth',2);
    xlabel('Number of Trees','FontSize',30)
    ylabel('Out of Bag Error','FontSize',30)
    title(['Out of Bag Error, iteration ' num2str(iter)],'FontSize',30)
    set(gca,'FontSize',20)
    set(gca,'LineWidth',2);   
    grid on
    drawnow
    fn=['error_number_trees_' num2str(iter)];
    wrplotepspng([PlotsDir fn]);


    %--------------------------------------------------------------------------
    %plot scater plot with error bars
    resTrain=fit_t-OutTrain;
    resTest=fit-OutTest;
    
    presTrain=100*resTrain./OutTrain;
    presTest=100*resTest./OutTest;
    
    %--------------------------------------------------------------------------
    figure('Position',[left, bottom, width, height],'Renderer',Renderer)

    %--------------------------------------------------------------------------
    jfigure=jfigure+1;
%     left=left+width;
%     if left>4*width
%         left=1;
%         %bottom=bottom+height;
%         if bottom>4*height
%             bottom=1;
%         end
%     end

    %--------------------------------------------------------------------------
    edges=-5:0.05:5;
    
    histogram(resTrain,edges);
    hold on
    histogram(resTest,edges);
    hold off
    
    grid on
    if iter <=1
        xlim([-4 4])
    elseif iter==2
        xlim([-1 1])
    else
        xlim([-.5 .5])
        xt=-0.5:0.1:0.5;
        set(gca,'XTick',xt);    
    end
    set(gca,'FontSize',20)
    set(gca,'TickDir','out');
    legend('Training','Validation')
    xlabel('Residual','FontSize',30)
    ylabel('Counts','FontSize',30)                    
    title([description ' (R is ' num2str(this_r,2) ') # ' num2str(iter)],'FontSize',20)
    drawnow

    fn=['residual_' num2str(iter)];
    wrplotepspng([PlotsDir fn]);
    
    %--------------------------------------------------------------------------
    figure('Position',[left, bottom, width, height],'Renderer',Renderer)

    %--------------------------------------------------------------------------
    jfigure=jfigure+1;
    left=left+width;
    if left>4*width
        left=1;
        %bottom=bottom+height;
        if bottom>4*height
            bottom=1;
        end
    end
    
    %--------------------------------------------------------------------------
    pedges=-40:0.2:40;
    
    histogram(presTrain,pedges);
    hold on
    histogram(presTest,pedges);
    hold off
    
    grid on
    if iter <=1
        xlim([-40 40])
    elseif iter <=2
        xlim([-10 10])
    elseif iter <=3
        xlim([-5 5])        
        xt=-5:1:5;
        set(gca,'XTick',xt);  
    elseif iter <=7
        xlim([-2.5 2.5])        
    else
        xlim([-2 2])
    end
    set(gca,'FontSize',20)
    set(gca,'TickDir','out');
    legend('Training','Validation')
    xlabel('% Residual','FontSize',30)
    ylabel('Counts','FontSize',30)                    
    title([description ' (R is ' num2str(this_r,2) ') # ' num2str(iter)],'FontSize',20)
    drawnow

    fn=['presidual_' num2str(iter)];
    wrplotepspng([PlotsDir fn]);
        
    
    %--------------------------------------------------------------------------
    % Calculate the relative importance of the variables
    tic
    disp('Calculating the relative importance of the input variables')
    oobErrorFullX = b.oobError;
    toc

    tic
    disp('Sorting importance into descending order')
    err=b.OOBPermutedVarDeltaError;
    [B,ierr] = sort(err,'descend');
    toc

    %--------------------------------------------------------------------------
    ntop=min([20 length(ierr)]);
    n_teir1=min([5 length(ierr)]);
    n_teir2=min([10 length(ierr)]);
    
    %--------------------------------------------------------------------------
    figure('Position',[left, bottom, width, height],'Renderer',Renderer)

    %--------------------------------------------------------------------------
    jfigure=jfigure+1;
    left=left+width;
    if left>4*width
        left=1;
        %bottom=bottom+height;
        if bottom>4*height
            bottom=1;
        end
    end

    %--------------------------------------------------------------------------
    %plot relative importance rank
    barh(err(ierr(1:ntop)),'g');
    xlabel('Variable Importance','FontSize',12,'Interpreter','latex');
    ylabel('Variable Rank','FontSize',12,'Interpreter','latex');
    title(['Relative Importance of Inputs, iteration ' num2str(iter)],...
        'FontSize',12,'Interpreter','latex'...
        );

    %--------------------------------------------------------------------------


    %--------------------------------------------------------------------------
    % Add text labels to each bar
    for ii=1:ntop
        text(...
            max([0 err(ierr(ii))+0.01*max(err)]),ii,...
            strrep(AllNamesWeights{ierr(ii)},'_',''),'Interpreter','latex','FontSize',11);
    end

    %--------------------------------------------------------------------------
    hold on
    barh(err(ierr(1:n_teir2)),'y');
    barh(err(ierr(1:n_teir1)),'r');
    hold off

    %--------------------------------------------------------------------------
    set(gca,'FontSize',20)
    set(gca,'TickDir','out');
    set(gca, 'ydir', 'reverse')
    set(gca,'LineWidth',2);   
    drawnow        
    
    grid on 
    xt = get(gca,'XTick');    
    xt_spacing=unique(diff(xt));
    xt_spacing=xt_spacing(1);    
    yt = get(gca,'YTick');    
    ylim([0.25 ntop+0.75]);
    xl=xlim;
    xlim([0 1.25*max(err)]);
    
    fn=['ranking_' num2str(iter)];
    wrplotepspng([PlotsDir fn]);            

    InTest=[InTest fit];
    InTrain=[InTrain fit_t];
    AllNamesWeights{end+1}=['Fit' num2str(iter)];    
    
    
    fit_err=OutTest-fit;
    fit_err_t=OutTrain-fit_t;
    
    clear b
    
    description=['Random Forest Error with ' num2str(ntrees) ' trees'];
    disp(description)    
    b=TreeBagger(...
        ntrees,InTrain,fit_err_t,...
        'Method','regression',...
        'oobvarimp','on',...
        'minleaf',leaf,...
        'FBoot',fboot,...
        'Options',paroptions...        
        );
    
    mdlRF_err{iter}=b;

    err=predict(b,InTest);
    err_t=predict(b,InTrain);
    
    InTest=[InTest err];
    InTrain=[InTrain err_t];
    AllNamesWeights{end+1}=['Error' num2str(iter)];        
    
    fit=fit+fit_err;    
    fit_t=fit_t+fit_err_t;
    
    InTest=[InTest fit];
    InTrain=[InTrain fit_t];
    AllNamesWeights{end+1}=['UpdatedFit' num2str(iter)];      
    

    
    %--------------------------------------------------------------------------
    figure('Position',[left, bottom, width, height],'Renderer',Renderer)

    %--------------------------------------------------------------------------
    jfigure=jfigure+1;
    left=left+width;
    if left>4*width
        left=1;
        %bottom=bottom+height;
        if bottom>4*height
            bottom=1;
        end
    end

    %--------------------------------------------------------------------------
    plot(OutAll,OutAll,'-b','LineWidth',10)
    hold on   
    errorbar(OutTrain,fit_t,fit_err_t,'og')
    errorbar(OutTest,fit,fit_err,'or')    
    plot(OutAll,OutAll+100,':c','LineWidth',.1)
    plot(OutAll+100,OutAll,':c','LineWidth',.1)    
    plot(OutAll,OutAll+50,':c','LineWidth',.1)
    plot(OutAll+50,OutAll,':c','LineWidth',.1)       
    hold off
    grid on
    xlim([0 max(OutTest)])
    ylim([0 max(OutTest)])
    xlabel('Observed','FontSize',30)
    ylabel('Estimated','FontSize',30)                    
    title(['Random forest (R_T=' num2str(this_t,2) ', R_v=' num2str(this_r,2)  ') Iteration # ' num2str(iter)],'FontSize',15)
    legend('1:1','Training','Validation','Location','bestoutside')
    set(gca,'FontSize',20)
    
    %xlim([0 100]);ylim([0 100]);

    drawnow

    fn=['scatter_' num2str(iter)];
    wrplotepspng([PlotsDir fn]);
        
    
    

end
% end iteration loop

%--------------------------------------------------------------------------
figure('Position',[left, bottom, width, height],'Renderer',Renderer)

%--------------------------------------------------------------------------
jfigure=jfigure+1;
left=left+width;
if left>4*width
    left=1;
    %bottom=bottom+height;
    if bottom>4*height
        bottom=1;
    end
end

%--------------------------------------------------------------------------
plot(r_iter_t,'LineWidth',4)
hold on
plot(r_iter_v,'-r','LineWidth',4)
hold off

legend('Training','Validation','Location','southeast')

set(gca,'FontSize',20)
set(gca,'TickDir','out');
set(gca,'LineWidth',2);   
xlabel('Iteration','FontSize',30,'Interpreter','latex');
ylabel('Correlation Coefficient, R','FontSize',20,'Interpreter','latex');
title(['Correlation Coefficient as a function of Iteration'],'FontSize',20)
grid on
axis tight
drawnow  

fn=['r_with_iterations'];
wrplotepspng([PlotsDir fn]);

%--------------------------------------------------------------------------
figure('Position',[left, bottom, width, height],'Renderer',Renderer)

%--------------------------------------------------------------------------
jfigure=jfigure+1;
left=left+width;
if left>4*width
    left=1;
    %bottom=bottom+height;
    if bottom>4*height
        bottom=1;
    end
end

%--------------------------------------------------------------------------


plot(OutAll,OutAll,'-b','LineWidth',10)
hold on   
errorbar(OutTrain,fit_t,fit_err_t,'og')
errorbar(OutTest,fit,fit_err,'or')
plot(OutAll,OutAll+100,':c','LineWidth',.1)
plot(OutAll+100,OutAll,':c','LineWidth',.1)    
plot(OutAll,OutAll+50,':c','LineWidth',.1)
plot(OutAll+50,OutAll,':c','LineWidth',.1)       
hold off
grid on
% xlim([0 max([max(fit_t) max(fit)])])
% ylim([0 max([max(fit_t) max(fit)])])
% xlim([0 3450])
% ylim([0 3450])
xlim([0 max(OutTest)])
ylim([0 max(OutTest)])

xlabel('Observed','FontSize',30)
ylabel('Estimated','FontSize',30)                    
title(['Random forest (R_T=' num2str(this_r,2) ', R_v=' num2str(this_t,2)  ')'],'FontSize',30)
    legend('1:1','Training','Validation','Location','bestoutside')
set(gca,'FontSize',20)
%axis tight


drawnow

fn=['scatter_with_errorbars'];
wrplotepspng([PlotsDir fn]);


%--------------------------------------------------------------------------
fn_mat=['mdlRF-' num2str(length(AllNamesWeights)) '.mat'];
command=['save(' '''' fn_mat '''' ',' '''' 'AllNamesWeights' '''' ',' '''' 'mdlRF' '''' ',' '''' 'mdlRF_err' '''' ',' '''' '-v7' '''' ');'];
disp(command);
tic;eval(command);toc




%--------------------------------------------------------------------------
