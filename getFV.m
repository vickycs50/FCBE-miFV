function [tr_fv,tst_fv,op,codes] = getFV(train_bags,train_bag_labels,test_bags,test_bag_labels,opt,ext_bags,ops,codes,type)

            if type =="fv"
                num_train_bag = size(train_bags,1);
                num_test_bag = size(test_bags,1);

                train_instances =[];%zeros();
                 for ii = 1:num_train_bag
                     bag = train_bags{(ii),1};
                     train_instances = [train_instances; bag];

                 end
                 % create ode book
    %              opt=[];
    %              opt.kmeans_num_center = 1;
    %              opt.PCA_energy = 0.9;
                 flag=1;
                if isempty(codes)
                    while flag==1
                        [codes, opt] = CreateGMMCodebook([train_instances;cell2mat(ext_bags)],opt);
                        if sum(sum(codes.kmeans)~=0)== size(codes.kmeans,2)
                            flag=0;
                        end
                    end
                else
                    codes = codes;
                    opt = ops;
                end

                 op=opt;
                 train_instances;




                 % Convert Training data into FV format
                dim = opt.PCA_dim * opt.kmeans_num_center * 2;
                tr_fv = zeros(num_train_bag,dim);
                tr_labels = zeros(num_train_bag,1);
                for ii = 1:num_train_bag
                    tr_fv(ii,:) = ExtractFV(train_bags{ii,1},opt,codes);
                    tr_fv(ii,:) = tr_fv(ii,:) ./ norm(tr_fv(ii,:));
    %                 tr_labels(ii) = train_bag_labels{ii};
                end

                %Min-Max Normalization on Training
                minv = min(tr_fv);
                maxv = max(tr_fv) - minv;
                maxv = maxv +eps;
                maxv = 1./maxv;
                tr_fv = (tr_fv -repmat(minv,num_train_bag,1)) .* repmat(maxv,num_train_bag,1);
                if sum(sum(isnan(tr_fv))) >=1
                    tr_fv;
                end
                tr_fv = sparse(tr_fv);
                if sum(isnan(tr_fv))>=1
                    tr_fv;
                end





                % Convert Test data into FV format

                dim = opt.PCA_dim * opt.kmeans_num_center * 2;
                tst_fv = zeros(num_test_bag,dim);
                tst_labels = zeros(num_test_bag,1);
                for ii = 1:num_test_bag

                    tst_fv(ii,:) = ExtractFV(test_bags{ii,1},opt,codes);
                    tst_fv(ii,:) = tst_fv(ii,:) ./ norm(tst_fv(ii,:));
                    tst_labels(ii) = test_bag_labels{ii};
                end
                % Min-Max Normalization Training Data
                minv = min(tst_fv);
                maxv = max(tst_fv) - minv;
                maxv = maxv +eps;
                maxv = 1./(maxv+eps);
                tst_fv = (tst_fv -repmat(minv,num_test_bag,1)) .* repmat(maxv,num_test_bag,1);
                tst_fv = sparse(tst_fv);   

                if sum(sum(isnan(tst_fv))) >=1
                    tst_fv;
                end
                 tst_fv = sparse(tst_fv);
                 
            else
                
                 num_train_bag = size(train_bags,1);
                num_test_bag = size(test_bags,1);

                train_instances =[];%zeros();
                 for ii = 1:num_train_bag
                     bag = train_bags{(ii),1};
                     train_instances = [train_instances; bag];

                 end
                 % create ode book
    %              opt=[];
    %              opt.kmeans_num_center = 1;
    %              opt.PCA_energy = 0.9;
                if isempty(codes)
                    [codes, opt] = CreateKmeansCodebook([train_instances;cell2mat(ext_bags)],opt);
                else
                    codes = codes;
                    opt = ops;
                end

% % % % 
% % % %                  flag=1;
% % % %                 if isempty(codes)
% % % %                     while flag==1
% % % %                         [codes, opt] = CreateKmeansCodebook([train_instances;cell2mat(ext_bags)],opt);
% % % %                         if sum(sum(codes.kmeans)~=0)== size(codes.kmeans,1)
% % % %                             flag=0;
% % % %                         end
% % % %                     end
% % % %                 else
% % % %                     codes = codes;
% % % %                     opt = ops;
% % % %                 end

                 op=opt;
                 train_instances;




                 % Convert Training data into FV format
                dim = opt.PCA_dim * opt.kmeans_num_center ;
                tr_fv = zeros(num_train_bag,dim);
                tr_labels = zeros(num_train_bag,1);
                for ii = 1:num_train_bag
                    tr_fv(ii,:) = ExtractVLAD(train_bags{ii,1},opt,codes);
                    tr_fv(ii,:) = tr_fv(ii,:) ./ norm(tr_fv(ii,:));
                    
% % % %                       tr_fv(ii,:) = encodevlad(codes.kmeans, train_bags{ii,1}*codes.lf);
                   
    %                 tr_labels(ii) = train_bag_labels{ii};
                end

                %Min-Max Normalization on Training
                minv = min(tr_fv);
                maxv = max(tr_fv) - minv;
                maxv = maxv +eps;
                maxv = 1./maxv;
                tr_fv = (tr_fv -repmat(minv,num_train_bag,1)) .* repmat(maxv,num_train_bag,1);
                if sum(sum(isnan(tr_fv))) >=1
                    tr_fv;
                end
                tr_fv = sparse(tr_fv);






                % Convert Test data into FV format

                dim = opt.PCA_dim * opt.kmeans_num_center;
                tst_fv = zeros(num_test_bag,dim);
                tst_labels = zeros(num_test_bag,1);
                for ii = 1:num_test_bag

                    tst_fv(ii,:) = ExtractVLAD(test_bags{ii,1},opt,codes);
                    tst_fv(ii,:) = tst_fv(ii,:) ./ norm(tst_fv(ii,:));
%                     tst_labels(ii) = test_bag_labels{ii};
%                      tst_fv(ii,:) = encodevlad(codes.kmeans, test_bags{ii,1}*codes.lf);
                end
%                 tst_fv = tst_fv ./ norm(tst_fv);
                % Min-Max Normalization Training Data
                minv = min(tst_fv);
                maxv = max(tst_fv) - minv;
                maxv = maxv +eps;
                maxv = 1./(maxv);
                tst_fv = (tst_fv -repmat(minv,num_test_bag,1)) .* repmat(maxv,num_test_bag,1);
                tst_fv = sparse(tst_fv);   

                if sum(sum(isnan(tst_fv))) >=1
                    tst_fv;
                end
                 tst_fv = sparse(tst_fv);
            end
            


if(sum(isnan(tr_fv)>=1))
    tr_fv;
end

end

