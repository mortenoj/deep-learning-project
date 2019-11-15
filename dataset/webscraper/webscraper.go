package webscraper

import (
	"fmt"
	"strings"
	"time"

	"github.com/gocolly/colly"
	"github.com/sirupsen/logrus"
)

// GetLinks scrapes hltv.org for download demo url
func GetLinks(offset int) ([]string, error) {
	var (
		err     error
		baseURL = "https://www.hltv.org/results"
		url     = fmt.Sprintf("%s?offset=%d&content=demo", baseURL, offset)
	)

	c := colly.NewCollector()
	linkCollector := c.Clone()

	limitRule := &colly.LimitRule{
		DomainGlob:  ".*hltv.*",
		Delay:       1000 * time.Millisecond,
		RandomDelay: 1000 * time.Millisecond,
	}

	err = linkCollector.Limit(limitRule)
	if err != nil {
		return nil, err
	}

	return crawl(url, c, linkCollector)
}

func crawl(url string, c, linkCollector *colly.Collector) ([]string, error) {
	var (
		err     error
		links   []string
		visited []string
		domain  = "https://www.hltv.org"
		exit    bool
	)

	handleErrors(c)
	//handleErrors(linkCollector)

	linkCollector.OnRequest(func(r *colly.Request) {
		visited = append(visited, r.URL.String())
	})

	linkCollector.OnHTML(".stream-box", func(e *colly.HTMLElement) {
		band, ok := e.DOM.Find("a").Attr("href")
		if ok && band != "" {
			links = append(links, domain+band)
		}
	})

	c.OnHTML(".result-con", func(e *colly.HTMLElement) {
		band, ok := e.DOM.Find("a").Attr("href")
		if !ok || band == "" || !strings.Contains(band, "/matches/") {
			return
		}

		if contains(visited, domain+band) || exit {
			return
		}

		err = linkCollector.Visit(domain + band)
		if err != nil {
			if err.Error() == "Too Many Requests" {
				logrus.Error("Blocked for too many requests. Try again later")
				exit = true
				return
			}
			panic(err)
		}
	})

	err = c.Visit(url)
	if err != nil {
		return nil, err
	}

	return links, nil
}

func handleErrors(c *colly.Collector) {
	c.OnError(func(r *colly.Response, err error) {
		fmt.Println("Request URL:", r.Request.URL, "failed with response:", string(r.Body), "\nError:", err)
	})
}

func contains(arr []string, elem string) bool {
	for _, e := range arr {
		if e == elem {
			return true
		}
	}

	return false
}
