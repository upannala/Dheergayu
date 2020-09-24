import requests

url = "http://3.16.163.12:5010/api/models/sleep"
r= requests.post(url,json={
	"TST":972341,
	"TIB":2.13424,
	"SE":0.234232,
	"W":0.230614,
	"S1":0.230614,
	"S2":0.230614,
	"S3":0.230614,
	"REM_Density":1.14,
	"REM":0.22151
})

print(r.json())