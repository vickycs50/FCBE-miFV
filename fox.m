
clear all
run('vlfeat-0.9.18\toolbox\vl_setup.m');
addpath('data/');
addpath('FSASL-master/')
addpath('SubspaceLearning/')
addpath('../data/figure');
addpath('../data/musk');
dataset_name = 'data/fox.mat';
load(dataset_name);
num_bag = size(data,1);
cluster={};

% load(str);
%%
num_fold = 1;
num_CV = 10;
models={};


acc = zeros(num_fold,num_CV);
acc2 = zeros(num_fold,num_CV);

val_proj=[];
tst_proj=[];
auc=[];
fi=1;
temp_set=[];
pca=0.85;
kn=[1];  
for i=1:10 % num of repeatation

      indices = crossvalind('Kfold',size(data,1),10);
 
    for j =1:10  %num_fold         
        load('data/figure_testIndex.mat');
        cur_testIndex = testIndex((i-1)*num_CV+j,:);

        cur_trainIndex = 1:num_bag;
        cur_trainIndex(cur_testIndex) = [];
        num_train_bag = size(cur_trainIndex,2);
        num_test_bag = size(cur_testIndex,2);
        num_train_bag = size(cur_trainIndex,2);
        num_test_bag = size(cur_testIndex,2);
        train_bags = data(cur_trainIndex,1);
        train_bag_labels = data (cur_trainIndex,2);
        test_bags = data(cur_testIndex,1);
        test_bag_labels = data(cur_testIndex , 2);        
         
        train_instances =[];
        train_lable=[];
                 
                 %extract instances for clusstering
        Xtb=[];
        num_train_bag = size(train_bags,1);
        for ii = 1:num_train_bag
       
             train_instances = [train_instances; train_bags{ii,1}]; %#ok<AGROW>
             for jj=1:size(train_bags{ii,1},1)
                         
                     train_lable = [train_lable; train_bag_labels{ii}];
                     Xtb = [Xtb;cur_trainIndex(ii)];
             end

         end
          D.X=train_instances;
          D.Y= train_lable;
          D.B = cur_trainIndex';
          D.YB = cell2mat(train_bag_labels);
          D.XtB = Xtb;
          D.YR = train_lable;
                    
          nSel =2;
          nK = 5;    % number of clusters in k-means
          nRS =15;   % number of random subspaces
          nDSS = 0.05;   % number of dimension per subspace
          T = 0.0100;     
          
          
          
          res = zeros(nRS,size(D.X,1));

          nDSS = ceil(size(D.X,2)*nDSS);
          pop=[];
          ens_label=[];
          ens_yval=[];
          ens_label2=[];
          ens_yval2=[];
          for kk=1:10
                for s = 1:nRS
                        % Randomly select the dimension for the subspace
                        ind = randperm(size(D.X,2));
                        ind = ind(1:nDSS);
                        R = false(size(D.X,2), 1);
                        R(ind) = true;
                         % Cluster data in the subspace
                        [C, A] = vl_kmeans(D.X(:,R)', nK);
                         C = C';
                         % compute proporition of positive bag per cluster
                        pC = zeros(size(C,1),1);
                        ctnC = zeros(size(C,1),1);
                         options = [1.5 NaN NaN 0];
                        [centers,U] = fcm(D.X(:,R),nK,options);
                         C = centers;
                         C=centers;
                         U_prob = sum(U(:,1:max(find(D.Y==1))),2)./sum(U,2);
                         U_prob2 = sum(U(:,max(find(D.Y==1))+1:end),2)./sum(U,2);
                         prob = [];
                         for cc=1:size(U,2)
                             prob(cc,1)=sum(U(:,cc).*U_prob);
                         end
                        res(s,:) = prob';
                 end
                score = mean(res,1);
                probX = score*0;
                %% create softmax probability vector
                bagList = unique(D.XtB);
                for ii = 1:length(bagList)

                    % get denuminator
                    idb = (D.XtB == bagList(ii));
                    denum = sum(exp(score(idb)/T));

                    % get index of elements in the bag
                    idi = 1:length(D.XtB);
                    idi = idi(idb);

                    for jj = 1:length(idi)
                        probX(idi(jj)) = exp(score(idi(jj))/T)/denum;
                    end

                end
               
             selection = false(nSel,size(D.X,1));
                for jj = 1:nSel

                        % select one instance per bag
                        for ii = 1:length(bagList)
%                            
                            % get probabilities of instance of the bag
                            idb = (D.XtB == bagList(ii));
                            p = probX(idb);

                            % get index of elements in the bag
                            idi = 1:length(D.XtB);
                            idi = idi(idb);

                            % get index of the selected instance
                           
                            cumm = 0;
                             ind=[];                               
                                       
                     
                     
                     if data{ bagList(ii),2}==0
                                [B,I] = sort(p);
                     end
                            
                            
                            p = p(I);
                            idi = idi(I);
                             t=rand();
                            for k = 1:length(idi)
                                cumm = cumm + p(k);
                                ind = [ind;k];
                                if t < cumm
                                    break;
                                end
                            end

                           
                            selection(jj,idi(ind)) = true;

                        end
                end  

                pop=selection;
         
         

           % make new bags acording to selected instances
              models={};
              n_train_bags = {};
              for kkp=1:size(pop,1)
                 for ii = 1:num_train_bag
                          bag = train_bags{ii};
                                % get probabilities of instance of the bag
                                idb = (D.XtB == bagList(ii));
                                idi = 1:length(D.XtB);
                                idi = idi(idb);
                                ind = pop(kkp,idi);
                                bag = bag(logical(ind),:);
                                n_train_bags{ii,1}=bag;
                 end
    %              
         
                 n_test_bags = test_bags;                                 
                    opt.kmeans_num_center =kn
                  if pca==100
                      opt.PCA_energy = 0;
                  else
                     opt.PCA_energy = pca/100;
                  end
                            
                            
                            
                            
                            
                    nDFV=1;
                    nDFV = ceil(size(D.X,2)*nDFV);
                    ind = randperm(size(D.X,2));
                    ind = ind(1:nDFV);
                    R = false(size(D.X,2), 1);
                    R(ind) = true;
                    opt.R = R;                         
                    type = "fv";

                    % generate FV
                    [tr_fv,tst_fv,op,codes]=getFV(n_train_bags,train_bag_labels,n_test_bags,test_bag_labels,opt,[],[],[],type);

                    [val_fv,t_fv,~,~]=getFV(train_bags,train_bag_labels,n_test_bags,test_bag_labels,opt,[],op,codes,type);

                    %SRKDA base learner
                     params = [];
                     params.deg = 2;
                     params.b = 0;
                     params.gamma = 0.5;
                     params.beta = 0.5; % 1.25 for emotions, 1.3 for Yeast
                     params.p = 1;
                     t = sqdist(tr_fv',tr_fv');
                     A = mean(t);
                     A = mean(A);
                     params.gamma = 1.0./A;


                    option.KernelType = 'Gaussian';
                    option.ReguGamma =  params.gamma ;
                    option.ReguType = 'Ridge';

                    if isnan(params.gamma)==true
                            params.gamma
                    end


                     model = SRKDAtrain(tr_fv, cell2mat(train_bag_labels),option,tst_fv);                
                     project = model.projection;
                     val_proj = [val_proj,project];      
                     [feaNew] = SRKDAtest(tst_fv, model);
                     tst_proj = [tst_proj,feaNew];

                     [accuracy,pred_label] = SRKDApredict(tst_fv, cell2mat(test_bag_labels), model);
                     ens_label = [ens_label, pred_label];
                     [ accuracy, val_yhat]  = SRKDApredict(val_fv, cell2mat(train_bag_labels), model);
                     ens_yval = [ens_yval, val_yhat];
                            
                            
                            

                       % training SVM base learner

                      model = train(cell2mat(train_bag_labels),tr_fv,'-s 1 -c 0.05 -B -1 -q');
                      [pred_t, accuracy, dec_val] = predict(cell2mat(test_bag_labels),tst_fv,model);
                      ens_label = [ens_label, pred_t, dec_val];
                      [pred_v, accuracy, dec_val] = predict(cell2mat(train_bag_labels),val_fv,model);
                      ens_yval = [ens_yval, pred_v, dec_val];

                             
             
              end
            end
           
           ens_label1 =[ ens_label,tst_proj];
           ens_yval1 = [ens_yval,val_proj];
        
        %MEta learner 
          addpath('svm//')
         
          ens_yval2 = zscore(ens_yval1);
          ens_label2 = zscore(ens_label1);
        % normalization
                minv = min(ens_yval2);
                maxv = max(ens_yval2) - minv;
                maxv = maxv +eps;
                maxv = 1./maxv;
                ens_yval2 = (ens_yval2 -repmat(minv,num_train_bag,1)) .* repmat(maxv,num_train_bag,1);
          

         ens_label2 = (ens_label2 -repmat(minv,num_test_bag,1)) .* repmat(maxv,num_test_bag,1);
         model = train(cell2mat(train_bag_labels),sparse(ens_yval2),'-s 1 -c 0.05 -B -1 -q');
        [pred_t, accuracy, dec_val] = predict(cell2mat(test_bag_labels),sparse(ens_label2),model);
        acc(i,j)=accuracy(1);
       
         
        [X,Y,T,AUC] = perfcurve(cell2mat(test_bag_labels),sc,1);
        auc(i,j)=AUC;
 
    end
save('results');
end
disp(['Accuracy = ',num2str(mean(mean(acc))),'¡À',num2str(std(acc(:)))]);
