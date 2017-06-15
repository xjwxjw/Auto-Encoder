hidden_state = importdata('hidden_state.txt');
for i=1:512
     plot(hidden_state(1:1472,i));
     pause;
%      saveas(gcf,['./result/',num2str(i)],'jpg');
end
% plot(dis(1:1400,1));
