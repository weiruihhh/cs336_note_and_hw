我这里对fasttext的训练和分类的流程和讲义上的实现很不一样。
讲义上的思路是通过enwiki-20240420-extracted_urls.txt.gz (43.5M URLs) 的URL来下载对应的wiki文章，然后进行质量过滤，最后将过滤后的文章转换为fasttext的训练集格式。
但由于这个wget获取网页内容的速度实在太慢了，我放弃了这个思路。
我的做法是直接下了一个现成的整理好的enwiki_latest_pages_articles.xml.bz2，然后通过wikiextractor脚本提取出wiki文章的json格式，然后进行质量过滤，最后将过滤后的文章转换为fasttext的训练集格式。

wget <https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2>

git clone <https://github.com/attardi/wikiextractor.git>

还有一点要注意的是，要根据你自己设置的label来修改test_quality.py中的assert判断。比如我的正负样本标签是hq和lq，那么test_quality.py中的assert判断就要修改为assert prediction == "hq"和assert prediction == "lq"。
