import xlrd
import re
import jieba

jieba.load_userdict('./dict.txt')
def get_tcs(juji,times,j,sheet_num):
    print('+++++++++++++++++++++++++++++++++++++')
    if sheet_num==0:
        type=j
    else:
        type=j+10
    txt_path='e:/弹幕2/'+juji[0:-2]+'/'+str(int(juji[-2:]))+'.txt'
    print(txt_path)
    fr=open(txt_path,'r',encoding='utf-8')
    lines=fr.readlines()
    # print(txt_path,times,j)
    for i in times:
        start=i.split('~')[0]
        start_min=start.split(':')[0]
        start_sec=start.split(':')[1]
        st=int(start_min)*60+int(start_sec)
        end=i.split('~')[1]
        end_min=end.split(':')[0]
        end_sec = end.split(':')[1]
        ed=int(end_min)*60+int(end_sec)
        print(st,ed)
        #print(end,start)
        tcss=[]
        for line in lines[1:]:
            second=float(line.split('  ')[0])
            tcs=line.split('  ')[2].strip('\n')
            if second>=st and second<ed:
                tcss.append(str(second).split('.')[0]+' '+tcs)
                #print(str(second).split('.')[0]+' '+tcs)
        write_tcs(tcss,type)



def get_xlsx_data():
    workbook=xlrd.open_workbook(r'e:/弹幕2/highlight_not.xlsx')
    for i in range(0,2):
        print(str(i)+'===================================================')
        sheet=workbook.sheet_by_index(i)
        sheet_num=i
        nrows=sheet.nrows
        ncols=sheet.ncols
        print(nrows)
        for i in range(1,nrows):
            juji=sheet.cell_value(i,0)
            for j in range(1,ncols):
                times=[]
                cell_cont=sheet.cell_value(i,j)
                if cell_cont!='':
                    if '|' in cell_cont:
                        times=cell_cont.split('|')
                    else:
                        times=[cell_cont]
                #print(times)
                if times!=[]:
                    get_tcs(juji,times,j,sheet_num)



def write_tcs(tcss,type):
    print('##################################3')
    print(type,tcss)
    fw=open('./tcs_train.txt','a',encoding='utf-8')
    fw.write('type:'+str(type)+'\n')
    for i in tcss:
        time=i.split(' ')[0]
        cont=i.split(' ')[1]
        cont = re.sub('[\s+\.\!\/_,$%^*(+\"\')]+|[-；'
                   '✺◟❛ ั∗◞━ ゝ‵ □ ′◢ ☄丶━┻ ฺ┻♡💔♥❤💝💖💙💗💞💕❤ 💙️💝💓Ĺ̯̿̿▀↙\[\]–：🙃ლ╹௰Ő'
                   'мфрхсуъюьэыБяЖГЕА↑ЬкжёбифртцсхмБеэгплхой◡аСук💣🇳😄╯🇨★♂😂💘╰'
                   '〖〗۶げちゃ駄ぶりわ▼👲🐶😹✋りエヴァンゲリオン‘◔◡おノ°‵′♀≠ノジョ\\≧≦οり╯°︵ˊᵒ̴̶̷̤ꇴᵒ̴̶̷̤ˋ '
                   '⊃□⁽⁽ଘ⬅←↑↓→↘™┴’‵□′🎉👇👏━┻〃─΄✹ਊ‵┴ェォコオィォコ々ワムಥ￢﹏இ皿유전자「ぅぁぅぉおりっ′Ü☀ᆺ罒독일미국メリ◦˙'
                   ';⌓°≺ぞみԅひり¯ㅂㆁ＋|〔〕독일에는👣👧👦👨👩‵🌸□′👴👼👶🎅◕💂🏃🌶🐓🎀🍃💄♡🌱'
                   'ั ็ ็ ัั ็ ็ ั﹋͟͞ ͟͞◊ò ◊ óºゝ⬛■▲òó۶ಡ\≧➕≦⌄́ಡ√卍★■☆ヘ︶︿︶╮╯＿╰╭◔ ◡ ◔з」ᴗ✪`∇っ∑사대발견›‹👄¦❀☆ʘ🔥Őヾ☜🍎💑👑⚆↗💃▄█▀👹🏅≡👑🍗🍳🌶🍚🍲🍱🍛🍣🍥🍝🍕🍛🍟🍩🍫🍿🍪 '
                   '😖╥๓ね［］❆꒳ˋ̋ˊ;˘口❢＞┏┛﹃０∩🌽✘❓▓🐔🐴🐷😳👆♪👍 ⌣ ˘ღ🚶🏈🎎🇨🇳〃┯━🔪✈🍉♬♩♫♪💩❌⭕🎊🐜😎🕷◝‿◜😱🍇🍓🍈🍒﹀▂🍑🍠🖖🍞🌹🎤🎶🍯🍖🐍🍤👍🍳✿🙌🎎👄👉👌♀🐵👣🐾🌝😍😏👻🐽🐣🐛💋😡😨😉😱🌚🌝😧😈🇨🇳😪😌😘😞😍😏☻≡😁😝🌚😕😅😭😊😄😂㉨੭ ✧👍🍢˶'
                   '‾ ᷄――«°\\ヘˇ ‸ ˇ＾ワ∠ノπ∼へ'
                   '※艹0123456789qwertyuiopasdfhjklzxcvnQWERTYUIOPASDFHJKLZXCVN》'
                   'Ⅷ⊂・ㅅㅇｃｖろきＯ☞《•シざびすドほばパトかやてし ֟ょだどかめ🐛てだを％눈アスカゼテはようにそつづと'
                   '≖ナوθ『』﹁꒦ ີ゜\\{}きくれいたけならさえまもとこにんでるあたが\.\.\.＝/😔😗😛'
                   '＠モＯゞ＼⁄づ= - =×℃ ≈ρ～!_:เ ีﾉ＊…εᕪ · Д\\ ·> <ﾟ·!｀⊙¬のΣ дヽー'
                   '()灬ꈍ▽̀∀•́ ωฅ /○—●^*٩´∀ ง๑+——()▃?【】“”！，。？、~@#￥%……&*（）,：-]+', '', cont)
        if len(cont) <= 1:
            continue
        else:
            str_load = jieba.cut(cont)
            fw.write(time+'  '+' '.join(str_load) + '\n')
    fw.close()


get_xlsx_data()
root='e:/弹幕2'