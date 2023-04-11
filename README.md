第十四届蓝桥杯大赛软件赛省赛 Python 大学 B 组
Python 大学 B 组

试题 A: 2023
本题总分：5 分

【问题描述】

请求出在 12345678 至 98765432 中，有多少个数中完全不包含 2023 。 完全不包含 2023 是指无论将这个数的哪些数位移除都不能得到 2023 。 例如 20322175，33220022 都完全不包含 2023，而 20230415，20193213 则 含有 2023 (后者取第 1, 2, 6, 8 个数位) 。

【答案提交】

这是一道结果填空的题，你只需要算出结果后提交即可。本题的结果为一 个整数，在提交答案时只填写这个整数，填写多余的内容将无法得分。


![image](https://user-images.githubusercontent.com/76508404/230752954-245e087d-93db-4dbb-aa45-4e6ad7859050.png)

![image](https://user-images.githubusercontent.com/76508404/230753091-f12afaa8-3087-42a1-aafb-a1769aac7254.png)





试题 B: 硬币兑换
本题总分：5 分

【问题描述】

小蓝手中有 2023 种不同面值的硬币，这些硬币全部是新版硬币，其中第i(1 ≤ i ≤ 2023) 种硬币的面值为 i ，数量也为 i 个。硬币兑换机可以进行硬币兑 换，兑换规则为：交给硬币兑换机两个新版硬币 coin1 和 coin2 ，硬币兑换机会 兑换成一个面值为 coin1 + coin2 的旧版硬币。 小蓝可以用自己已有的硬币进行任意次数兑换，假设最终小蓝手中有 K 种 不同面值的硬币（只看面值，不看新旧）并且第 i(1 ≤ i ≤ K) 种硬币的个数为sumi。小蓝想要使得 max{sum1, sum2, · · · , sumK} 的值达到最大，请你帮他计算 这个值最大是多少。 注意硬币兑换机只接受新版硬币进行兑换，并且兑换出的硬币全部是旧版 硬币。

【答案提交】

这是一道结果填空的题，你只需要算出结果后提交即可。本题的结果为一 个整数，在提交答案时只填写这个整数，填写多余的内容将无法得分。


![image](https://user-images.githubusercontent.com/76508404/230753122-8fe8bc7e-3253-4915-879e-e5842bef9b31.png)


![image](https://user-images.githubusercontent.com/76508404/230753112-4ef3c0d3-19eb-42b1-b817-bb7d18ed4618.png)





试题 C: 松散子序列
时间限制: 10.0s 内存限制: 512.0MB 本题总分：10 分

【问题描述】

给定一个仅含小写字母的字符串 s ，假设 s 的一个子序列 t 的第 i 个字符 对应了原字符串中的第 pi 个字符。我们定义 s 的一个松散子序列为：对于 i > 1总是有 pi − pi−1 ≥ 2 。设一个子序列的价值为其包含的每个字符的价值之和 (a ∼ z 分别为 1 ∼ 26 ) 。 求 s 的松散子序列中的最大价值。

【输入格式】

输入一行包含一个字符串 s 。

【输出格式】

输出一行包含一个整数表示答案。

【样例输入】

azaazaz

【样例输出】

78

【评测用例规模与约定】

对于 20% 的评测用例，|s| ≤ 10 ； 对于 40% 的评测用例，|s| ≤ 300 ； 对于 70% 的评测用例，|s| ≤ 5000 ； 对于所有评测用例，1 ≤ |s| ≤ 10 6，字符串中仅包含小写字母。



![image](https://user-images.githubusercontent.com/76508404/230753173-4fe24f9a-ed67-437a-93d0-5afa386f8ea2.png)


![image](https://user-images.githubusercontent.com/76508404/230753179-52928d4a-8b96-4a29-86d2-c92545b042bc.png)








试题 D: 管道
时间限制: 10.0s 内存限制: 512.0MB 本题总分：10 分

【问题描述】

有一根长度为 len 的横向的管道，该管道按照单位长度分为 len 段，每一段 的中央有一个可开关的阀门和一个检测水流的传感器。 一开始管道是空的，位于 Li 的阀门会在 S i 时刻打开，并不断让水流入管 道。 对于位于 Li 的阀门，它流入的水在 Ti (Ti ≥ S i) 时刻会使得从第 Li−(Ti−S i)段到第 Li + (Ti − S i) 段的传感器检测到水流。 求管道中每一段中间的传感器都检测到有水流的最早时间。

【输入格式】

输入的第一行包含两个整数 n, len，用一个空格分隔，分别表示会打开的阀 门数和管道长度。 接下来 n 行每行包含两个整数 Li , S i，用一个空格分隔，表示位于第 Li 段 管道中央的阀门会在 S i 时刻打开。

【输出格式】

输出一行包含一个整数表示答案。

【样例输入】

3 10 

1 1

6 5

10 2

【样例输出】

5

【评测用例规模与约定】

对于 30% 的评测用例，n ≤ 200，S i , len ≤ 3000 ； 对于 70% 的评测用例，n ≤ 5000，S i , len ≤ 105 ； 对于所有评测用例，1 ≤ n ≤ 105，1 ≤ S i , len ≤ 109，1 ≤ Li ≤ len，Li−1 < Li。


![image](https://user-images.githubusercontent.com/76508404/230754496-fec19e78-d028-4c9d-880f-8a69b9a477d9.png)







试题 E: 保险箱
时间限制: 10.0s 内存限制: 512.0MB 本题总分：15 分

【问题描述】

小蓝有一个保险箱，保险箱上共有 n 位数字。 小蓝可以任意调整保险箱上的每个数字，每一次操作可以将其中一位增加1 或减少 1 。 当某位原本为 9 或 0 时可能会向前（左边）进位/退位，当最高位（左边第 一位）上的数字变化时向前的进位或退位忽略。 例如：

00000 的第 5 位减 1 变为 99999 ；

99999 的第 5 位减 1 变为 99998 ；

00000 的第 4 位减 1 变为 99990 ；

97993 的第 4 位加 1 变为 98003 ；

99909 的第 3 位加 1 变为 00009 。 保险箱上一开始有一个数字 x，小蓝希望把它变成 y，这样才能打开它，问 小蓝最少需要操作的次数。

【输入格式】

输入的第一行包含一个整数 n 。 第二行包含一个 n 位整数 x 。 第三行包含一个 n 位整数 y 。

【输出格式】

输出一行包含一个整数表示答案。


【样例输入】

5 

12349 

54321

【样例输出】

11

【评测用例规模与约定】

对于 30% 的评测用例，1 ≤ n ≤ 300 ； 对于 60% 的评测用例，1 ≤ n ≤ 3000 ； 对于所有评测用例，1 ≤ n ≤ 105，x, y 中仅包含数字 0 至 9，可能有前导零。

思路
求最少操作次数，用bfs。（刚开始用了dfs来做，浪费了些时间）

这个问题主要是要注意加减时的进位、借位。这里编写了op_minus，op_plus两个函数来进行处理。

然后因为加减时的进位借位都是右数（低位）向左数（高位）进位/借位，也就意味着会影响到下一位，所以我们需要先把低位先处理成目标字符，把它固定住。

然后bfs队列中的元素是源字符串src_old，目标字符串dst_old，当前操作次数cnt，当前要操作的位数idx。

然后只要发现源字符串和目标字符串相等了，就break掉循环。此时队内都是源字符串和目标字符串相等的元素，只有操作次数不相等，因此遍历一遍，取最小操作次数即可。


        from collections import deque
        import copy
        n = int(input())
        pwd_1 = input()
        pwd_2 = input()
        inf = 0x3f3f3f3f


        def op_plus(s, i):
            # 第i位进行加1操作
            tmp = int(s[i])
            if tmp == 9:
                tmp = 0
                if i != 0:
                    s = op_plus(s, i-1)  # 进行进位操作
            else:
                tmp += 1
            s = s[:i] + str(tmp) + s[i+1:]
            return s


        def op_minus(s, i):
            # 第i位进行减1操作
            tmp = int(s[i])
            if tmp == 0:
                tmp = 9
                if i != 0:
                    s = op_minus(s, i-1)
            else:
                tmp -= 1
            s = s[:i] + str(tmp) + s[i+1:]
            return s


        def bfs(src_old, dst_old, k):
            deq = deque([[src_old, dst_old, 0, k]])
            while len(deq) > 0:
                t = deq.popleft()
                src, dst, cnt, idx = t
                if src == dst:
                    break
                plus_cnt = copy.deepcopy(cnt)
                src_plus = copy.deepcopy(src)
                while src_plus[idx - 1] != dst[idx - 1]:
                    src_plus = op_plus(src_plus, idx - 1)
                    plus_cnt += 1
                deq.append([src_plus, dst, plus_cnt, idx-1])

                minus_cnt = copy.deepcopy(cnt)
                src_minus = copy.deepcopy(src)
                while src_minus[idx - 1] != dst[idx - 1]:
                    src_minus = op_minus(src_minus, idx - 1)
                    minus_cnt += 1
                deq.append([src_minus, dst, minus_cnt, idx-1])

            min_cnt = inf
            while len(deq) > 0:
                t = deq.popleft()
                src, dst, cnt, idx = t
                min_cnt = min(min_cnt, cnt)

            return min_cnt


        cnt = bfs(pwd_1, pwd_2, n)
        print(cnt)




试题 F: 树上选点
时间限制: 10.0s 内存限制: 512.0MB 本题总分：15 分

【问题描述】

给定一棵树，树根为 1，每个点的点权为 Vi 。 你需要找出若干个点 Pi，使得：

1. 每两个点 Px Py 互不相邻；

2. 每两个点 Px Py 与树根的距离互不相同；

3. 找出的点的点权之和尽可能大。 请输出找到的这些点的点权和的最大值。

【输入格式】

输入的第一行包含一个整数 n 。 第二行包含 n − 1 个整数 Fi ，相邻整数之间使用一个空格分隔，分别表示 第 2 至 n 个结点的父结点编号。 第三行包含 n 个整数 Vi，相邻整数之间使用一个空格分隔，分别表示每个 结点的点权。

【输出格式】

输出一行包含一个整数表示答案。

【样例输入】

5 

1 2 3 2

2 1 9 3 5

【样例输出】

11

【评测用例规模与约定】

对于 40% 的评测用例，n ≤ 5000 ； 对于所有评测用例，1 ≤ n ≤ 2 × 105，1 ≤ Fi < i，1 ≤ Vi ≤ 104 。



思路
题面的三个条件可以转化为：
一个点选了之后，父或子不能选。
同一层只能选一个
求选所有点的和的最大值
这样转化之后，问题就是很经典的选或不选问题。

对第一个条件，我们从上往下遍历，枚举每一层选或不选。

对于第二个条件，我们对一层选的时候求该层最大值即可。

由于一层只能选一个，那么进行层序遍历就可以了。

时间复杂度 O(n)，每个点只被遍历了一次。

代码
        from collections import defaultdict

        N = 2_000_10
        # 存放权值
        w = [0 for i in range(N)]
        # 存放父节点
        f = [0 for i in range(N)]
        # 存放点的高度
        h = [0 for i in range(N)]
        # 存放每一层的点
        h_g = defaultdict(list)
        dp = [0 for i in range(N)]

        h_g[1].append(1)

        n = int(input())
        f[2:n] = list(map(int, input().split()))
        f[1] = -1
        w[1:n] = list(map(int, input().split()))

        for i in range(2, n + 1):
            # 一个点的高度是他父亲的高度加一
            h[i] = h[f[i]] + 1
            # 为了方便遍历，直接将这个点加入他所在的层
            h_g[h[i] + 1].append(i)

        dp[1] = w[1]
        # 从第二层遍历到最后一层
        for i in range(2, max(h[1:n]) + 1):
            i_max = max([w[j] for j in h_g[i]])
            dp[i] = max(dp[i - 2] + i_max, dp[i - 1])

        print(dp[max(h[1:n])])










试题 G: T 字消除
时间限制: 10.0s 内存限制: 512.0MB 本题总分：20 分

【问题描述】

小蓝正在玩一款游戏，游戏中有一个 n × n 大小的 01 矩阵 Ai, j 。 小蓝每次需要选择一个 T 字型的区域，且这个区域内至少要有一个 1 。选 中后，这个区域内所有的元素都会变成 0 。 给定游戏目前的矩阵，小蓝想知道他最多可以进行多少次上述操作。

T 字型区域是指形如 (x − 1, y)(x, y)(x + 1, y)(x, y + 1) 的四个点所形成的区 域。其旋转 90, 180, 270 度的形式同样也视作 T 字形区域。

【输入格式】

输入包含多组数据。 输入的第一行包含一个整数 D 表示数据组数。 对于每组数据，第一行包含一个整数 n 。 接下来 n 行每行包含 n 个 0 或 1，表示矩阵 Ai, j 的每个位置的值。

【输出格式】

输出 D 行，每行包含一个整数表示小蓝最多可以对当前询问中的矩阵操作 的次数。

【样例输入】

1 
3 
001
011
111

【样例输出】

5

【样例说明】

我们用 X 表示某次操作选中的 T 字形，以下给出一种可行方案：

001
XXX 
0X0 
00X 0X0 X00 011 => 0X1 => XXX => 0XX => XX0 => XX0 111 111 111 11X 1X0 X00

【评测用例规模与约定】

对于 10% 的评测用例，n = 3 ； 对于 40% 的评测用例，n ≤ 30 ； 对于所有评测用例，3 ≤ n ≤ 2000，矩阵中仅含 0 和 1 。



试题 H: 独一无二
时间限制: 30.0s 内存限制: 512.0MB 本题总分：20 分

【问题描述】

有一个包含 n 个点，m 条边的无向图，第 i 条边的边权为 ci，没有重边和 自环。设 si 表示从结点 1 出发到达结点 i 的最短路的不同路径数 ( i ∈ [1, n] )， 显然可以通过删除若干条边使得 si = 1，也就是有且仅有一条从 1 到 i 的最短 路，且保持最短路的路径长度不变，对于每个 i ，求出删除边数的最小值。

【输入格式】

输入的第一行包含两个正整数 n, m。 接下来 m 行，每行包含三个正整数 ui , vi , ci 表示第 i 条边连接的两个点的 编号和边权。

【输出格式】

输出 n 行，第 i 行包含一个正整数表示对于结点 i ，删除边数的最小值， 如果 1 和 i 不连通，输出 −1 。

【样例输入】

4 4

1 2 1

1 3 2

2 4 2

3 4 1

【样例输出】

0 

0

0

1


【样例说明】

在给定的图中，只有 s4 一开始为 2，因为有两条最短路：1 → 2 → 4, 1 →3 → 4，任意删掉一条边后，就可以只剩一条最短路。

【评测用例规模与约定】

对于 30% 的评测用例，n ≤ 1000； 对于所有评测用例，n ≤ 105 ，0 ≤ m ≤ min{ n(n−1) 2 , 106 } ，1 ≤ ui , vi ≤ n ，1 ≤ ci ≤ 10 。


思路
单源最短路，无负权边，首选就是dijkstra。画几个图稍微总结下规律就会发现只要每次出现相同距离的时候，任意删除一条边就可以得到正确的结果，属于贪心的思想。

代码
        import heapq

        N = 10
        g = {}
        inf = 0x3f3f3f3f
        st = [0 for i in range(N)]
        dist = [inf for i in range(N)]

        cnt = [0 for i in range(N)]


        def dijkstra():
            dist[1] = 0
            q = []
            heapq.heappush(q, (0, 1))
            while q:
                dis, cur = heapq.heappop(q)
                if st[cur] == 1:
                    continue
                st[cur] = 1
                if cur in g:
                    for c, d in g[cur]:
                        if d + dist[cur] == dist[c]:
                            cnt[c] += 1
                        if d + dist[cur] < dist[c]:
                            dist[c] = d + dist[cur]
                            heapq.heappush(q, (dist[c], c))


        n, m = map(int, input().split())
        for _ in range(m):
            a, b, c = map(int, input().split())
            if a in g:
                g[a].append((b, c))
            else:
                g[a] = [(b, c)]

        dijkstra()

        for i in range(1, n + 1):
            if dist[i] == inf:
                cnt[i] = -1
            print(cnt[i])











试题 I: 异或和

时间限制: 15.0s 内存限制: 512.0MB 本题总分：25 分

【问题描述】

给一棵含有 n 个结点的有根树，根结点为 1 ，编号为 i 的点有点权 ai（i ∈ [1, n]）。现在有两种操作，格式如下：

• 1 x y 该操作表示将点 x 的点权改为 y 。

• 2 x 该操作表示查询以结点 x 为根的子树内的所有点的点权的异或和。 现有长度为 m 的操作序列，请对于每个第二类操作给出正确的结果。

【输入格式】

输入的第一行包含两个正整数 n, m ，用一个空格分隔。 第二行包含 n 个整数 a1, a2, ..., an ，相邻整数之间使用一个空格分隔。 接下来 n − 1 行，每行包含两个正整数 ui , vi ，表示结点 ui 和 vi 之间有一条 边。 接下来 m 行，每行包含一个操作。

【输出格式】

输出若干行，每行对应一个查询操作的答案。

【样例输入】

4 4

1 2 3 4

1 2

1 3

2 4

2 1

1 1 0

2 1

2 2

【样例输出】

4 

5 

6

【评测用例规模与约定】

对于 30% 的评测用例，n, m ≤ 1000； 对于所有评测用例，1 ≤ n, m ≤ 100000 ，0 ≤ ai , y ≤ 100000 ，1 ≤ ui , vi , x ≤ n。


a维护以x为根的子树异或和, prev维护每个结点的点权。
每次change只需要向上更改贡献即可。
正解应该是 dfs序+树状数组/线段树 维护

        import bisect
        import sys
        import copy
        from collections import deque, defaultdict
        import heapq
        from itertools import accumulate, permutations, combinations
        import math

        input = lambda: sys.stdin.readline().rstrip("\r\n")
        printf = lambda d: sys.stdout.write(str(d) + "\n")


        # INF = 0x3f3f3f3f3f3f


        # sys.setrecursionlimit(100000)


        def read():
            line = sys.stdin.readline().strip()
            while not line:
                line = sys.stdin.readline().strip()
            return map(int, line.split())


        def I():
            return int(input())


        def dfs(u):
            for v in g[u]:
                a[u] ^= dfs(v)
            return a[u]


        def change(u, x, y):
            if u == x:
                a[x] = a[x] ^ prev[x] ^ y
                # 注意要把点权也修改了，比赛的时候忘记了。。。。
                prev[x] = y
                return prev[x] ^ y
            for v in g[u]:
                a[u] ^= change(v, x, y)
            return 0


        n, m = read()
        a = list(read())
        a = [0] + a
        prev = copy.deepcopy(a)
        g = [[] for _ in range(n + 1)]
        for i in range(n - 1):
            u, v = read()
            g[u].append(v)

        dfs(1)

        for _ in range(m):
            op = list(read())
            if op[0] == 1:
                x, y = op[1], op[2]
                change(1, x, y)
            if op[0] == 2:
                x = op[1]
                print(a[x])






试题 J: 混乱的数组

时间限制: 10.0s 内存限制: 512.0MB 本题总分：25 分

【问题描述】

给定一个正整数 x，请找出一个尽可能短的仅含正整数的数组 A 使得 A 中 恰好有 x 对 i, j 满足 Ai > Aj 。 如果存在多个这样的数组，请输出字典序最小的那个。

【输入格式】

输入一行包含一个整数表示 x 。

【输出格式】

输出两行。 第一行包含一个整数 n ，表示所求出的数组长度。 第二行包含 n 个整数 Ai，相邻整数之间使用一个空格分隔，依次表示数组 中的每个数。

【样例输入】

3

【样例输出】

3 
3 2 1

【评测用例规模与约定】

对于 30% 的评测用例，x ≤ 10 ； 对于 60% 的评测用例，x ≤ 100 ； 对于所有评测用例，1 ≤ x ≤ 109 。

![image](https://user-images.githubusercontent.com/76508404/230754566-92dbc2eb-edea-49db-8caa-b4b2e9f2909e.png)

