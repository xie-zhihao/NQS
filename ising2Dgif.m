function [E,M] = ising2Dgif(T,n,m,J,plot_flag,N)
% 2D Ising model (Metropolis) with optional live animation or GIF export
% Signature matched to ising2D_jx:
%   T: temperature
%   n,m: lattice size (rows, cols)
%   J: coupling
%   plot_flag: 0=none, 1=live animation, 2=export GIF
%   N: scale of steps (total steps t = 1e4*N)
%
% Returns:
%   E: average energy density (per spin)
%   M: average absolute magnetization density (per spin)

%% -------------------- 参数（可按需调整） --------------------
t = 1e4 * N;                   % 总蒙特卡洛步数（越大越充分）
targetFrames = 240;            % 目标动画帧数（导出/实时均使用）
plotEvery = max(1, floor(t/targetFrames));  % 抽帧间隔
livePause  = 0.03;             % 实时动画每帧暂停（秒）
% gifName    = 'ising2D_ZX.gif';    % GIF 文件名（plot_flag==2 时生效）
% gifSeconds = 10;               % GIF 目标总时长（秒）
% === GIF 设置（仅当 plot_flag==2 时生效） ===
gifName    = fullfile(pwd,'ising2D_ZX.gif');  % 绝对路径：当前目录
gifSeconds = 20;                            % 目标总时长（秒）
targetFrames = 240;                         % 目标帧数
plotEvery = max(1, floor(t/targetFrames));  % 按总步数均匀取帧

firstFrame = true;                          % 首帧标记
if plot_flag==2
    if exist(gifName,'file'); delete(gifName); end  % 删除旧文件，避免误判
    fprintf('GIF 将保存到：%s\n', gifName);
end

% ------------------------------------------------------------

%% 初始自旋（±1 随机）
S = sign(rand(n,m) - 0.5);
S(S==0) = 1;

Nspins  = n*m;
Magnet  = sum(S,'all');                                   % 标量
% 能量：只计"右、下"邻居避免双计
Energy  = -J * ( sum(S .* circshift(S,[0,1]), 'all') ...
               + sum(S .* circshift(S,[1,0]), 'all') );

% 记录序列
Elist = zeros(t,1);
Mlist = zeros(t,1);

% 预生成随机翻转位置
ii = randi(n, t, 1);
jj = randi(m, t, 1);

%% 画布 / GIF 初始化
doPlot = (plot_flag==1 || plot_flag==2);
if doPlot
    f = figure('Color','w','Name','Ising 2D');
    ax = axes('Parent',f);
    imagesc(ax, S); axis(ax,'image'); axis(ax,'off');
    colormap(ax, [0 0 0.8; 1 1 1]);      % -1:黑, +1:白
    caxis(ax,[-1 1]);
    title(ax, sprintf('T=%.3f, J=%.3f, %dx%d', T, J, n, m));
    drawnow;

    firstFrame = true;
end

%% Metropolis 更新
for k = 1:t
    i = ii(k); j = jj(k);

    % 周期边界的四邻居求和
    ip = (i==n) * 1   + (i< n) * (i+1);
    im = (i==1) * n   + (i> 1) * (i-1);
    jp = (j==m) * 1   + (j< m) * (j+1);
    jm = (j==1) * m   + (j> 1) * (j-1);

    nnSum = S(im,j) + S(ip,j) + S(i,jm) + S(i,jp);

    % 翻转代价 ΔE = 2 J s_ij * (sum of neighbors)
    dE = 2 * J * S(i,j) * nnSum;

    % 接受率
    if dE <= 0 || rand <= exp(-dE/T)
        % 更新自旋、能量、磁化
        old = S(i,j);
        S(i,j) = -old;
        Energy = Energy + dE;

        % ΔM = 新-旧 = (-old) - old = -2*old
        Magnet = Magnet - 2*old;
    end

    % 记录
    Elist(k) = Energy / Nspins;
    Mlist(k) = abs(Magnet) / Nspins;

    % 动画/导出
    if doPlot && mod(k,plotEvery)==0
        set(get(ax,'Children'),'CData',S);
        title(ax, sprintf('step %d/%d | E=%.4f | |M|=%.4f', ...
              k, t, Elist(k), Mlist(k)));
        drawnow;

        if plot_flag == 1
            pause(livePause);
        else % plot_flag == 2
            fr = getframe(f);
            [A,map] = rgb2ind(frame2im(fr),256);
            if firstFrame
                imwrite(A,map,gifName,'gif','LoopCount',inf, ...
                        'DelayTime', gifSeconds/(t/plotEvery));
                firstFrame = false;
            else
                imwrite(A,map,gifName,'gif','WriteMode','append', ...
                        'DelayTime', gifSeconds/(t/plotEvery));
            end
        end
    end
end

%% 热化丢弃 + 平均
drop = min(50*Nspins, numel(Mlist));   % 防越界
Mlist(1:drop) = [];
Elist(1:drop) = [];

M = mean(Mlist);
E = mean(Elist);

%% 可选的时序图（仅实时动画模式下展示）
if plot_flag == 1
    figure('Color','w');
    subplot(2,1,1); plot(Elist,'LineWidth',1); grid on;
    xlabel('step'); ylabel('E/N'); title('Energy density');
    subplot(2,1,2); plot(Mlist,'LineWidth',1); grid on;
    xlabel('step'); ylabel('|M|'); title('Magnetization density');
end
end
