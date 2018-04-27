#!/usr/bin/python
import sys, urllib, re, json, socket, string
from bs4 import BeautifulSoup
socket.setdefaulttimeout(20)
item_dict = {}
try:
    for line in open(sys.argv[1]):
        fields = line.rstrip('\n').split('\t')
        tweetid = fields[0]
        userid = fields[1]
	#print userid
	#print tweetid
	tweet = None
        text = "Not Available"
        if item_dict.has_key(tweetid):
            text = item_dict[tweetid]
        else:
            try:
                f = urllib.urlopen('http://twitter.com/'+str(userid)+'/status/'+str(tweetid))
                html = f.read().replace("</html>", "") + "</html>"
                soup = BeautifulSoup(html)
                jstt   = soup.find_all("p", "js-tweet-text")
		tweets = list(set([x.get_text() for x in jstt]))
		#print tweets                
		                
		if(len(tweets)) > 1:#means there are more than one tweet being displayed (new twitter design)
			other_tweets = []
			
			cont   = soup.find_all("div", "content")
			for i in cont:
				o_t = i.find_all("p","js-tweet-text")
				other_text = list(set([x.get_text() for x in o_t]))
				other_tweets += other_text					
			tweets = list(set(tweets)-set(other_tweets))
			#print 'Other tweets\n'			
			#print other_tweets                
		        #print tweets
			#print '\n'        
			#continue
		
                text = tweets[0]
                item_dict[tweetid] = tweets[0]
                for j in soup.find_all("input", "json-data", id="init-data"):
                    js = json.loads(j['value'])
                    if(js.has_key("embedData")):
                        tweet = js["embedData"]["status"]
                        text  = js["embedData"]["status"]["text"]
                        item_dict[tweetid] = text
                        break
            except Exception:
		#print userid,tweetid
                continue
    
        if(tweet != None and tweet["id_str"] != tweetid):
                text = "This tweet has been removed or is not available"
                item_dict[tweetid] = "This tweet has been removed or is not available"
        text = text.replace('\n', ' ',)
        text = re.sub(r'\s+', ' ', text)
        print "\t".join(fields + [text]).encode('utf-8')
except IndexError:
    print 'Incorrect arguments specified (may be you didn\'t specify any arguments..'
    print 'Format: python [scriptname] [inputfilename] > [outputfilename]'
