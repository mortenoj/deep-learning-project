package main

import (
	"fmt"
	"os"
	"path/filepath"

	dem "github.com/markus-wa/demoinfocs-golang"
	events "github.com/markus-wa/demoinfocs-golang/events"
	"github.com/mortenoj/deep-learning-project/dataset/utils"
	"github.com/mortenoj/deep-learning-project/dataset/webscraper"
	"github.com/sirupsen/logrus"
)

// Run like this: go run print_kills.go
func main() {
	var err error

	links, err := webscraper.GetLinks(0, 1)
	if err != nil {
		panic(err)
	}

	logrus.Infof("Available links: %v", links)

	for _, downloadLink := range links {
		err = handleDemoLink(downloadLink)
		if err != nil {
			panic(err)
		}
	}
}

func handleDemoLink(demoLink string) error {
	rarPath := "demos/demo.rar"

	err := utils.DownloadDemo(demoLink, rarPath)
	if err != nil {
		return err
	}

	files, err := utils.UnRar(rarPath, "demos")
	if err != nil {
		return err
	}

	for _, file := range files {
		err = extractDataset(file)
		if err != nil {
			return err
		}
	}

	matches, err := filepath.Glob("demos/*.dem")
	for _, file := range matches {
		os.RemoveAll(file)
	}

	return nil
}

func extractDataset(filePath string) error {
	var err error

	f, err := os.Open(filePath)
	if err != nil {
		return err
	}
	defer f.Close()

	p := dem.NewParser(f)

	tScoreRoundStart := 0
	ctScoreRoundStart := 0
	// Register handler on kill events
	//p.RegisterEventHandler(func(e events.Kill) {
	//if e.PenetratedObjects > 0 && !e.IsHeadshot {
	//fmt.Printf("Wallbang: %s => %s \n", e.Killer, e.Victim)
	//} else if e.IsHeadshot && e.PenetratedObjects > 0 {
	//fmt.Printf("Wallbang Headshot: %s => %s \n", e.Killer, e.Victim)
	//}
	//})

	p.RegisterEventHandler(func(e events.RoundFreezetimeEnd) {
		gs := p.GameState()

		ct := gs.TeamCounterTerrorists()
		t := gs.TeamTerrorists()

		tScoreRoundStart = t.Score
		ctScoreRoundStart = ct.Score

		serverPlayers := ct.Members()
		for _, p := range t.Members() {
			serverPlayers = append(serverPlayers, p)
		}

		// TODO: Possibly make a struct of all weapons for putting this into a JSON that matches the JSON output from python
		for _, player := range serverPlayers {
			//fmt.Printf("%v \n", player.Weapons())
			//fmt.Printf("%v \n", player.RawWeapons)
			for _, weapon := range player.Weapons() {
				fmt.Printf("%v \n", weapon.String())
			}
		}

		//ctEquipment := ct.FreezeTimeEndEquipmentValue()
		//tEquipment := t.FreezeTimeEndEquipmentValue()

		//fmt.Printf("CT Equipment value: %d\n", ctEquipment)
		//fmt.Printf("CT Equipment value: %d\n", ct.CurrentEquipmentValue())
		//fmt.Printf("T Equipment value: %d\n", tEquipment)
		//fmt.Printf("T Equipment value: %d\n", t.CurrentEquipmentValue())
		//fmt.Printf("CT equipment value: %d\n", t.CurrentEquipmentValue())
	})

	// RoundEndOffical is the most accurate and gives what we want, but it does not give the result of the last round
	// However we can know who won by the team that had the highest Score last
	p.RegisterEventHandler(func(e events.RoundEndOfficial) {
		gs := p.GameState()

		if gs.IsMatchStarted() != true {
			return
		}
		ct := gs.TeamCounterTerrorists()
		t := gs.TeamTerrorists()

		if t.Score > tScoreRoundStart {
			fmt.Printf("Terrorists won the round, score is now: T:%d - CT: %d\n", t.Score, ct.Score)
		} else {
			fmt.Printf("Counter-Terrorists won the round, score is now: T:%d - CT: %d\n", t.Score, ct.Score)
		}

		// TODO:
		// Save round winner in a array or something e.g. 1 = T win, 0 = CT win
		// Save these to file when everything is done parsing
	})

	// Parse to end
	err = p.ParseToEnd()
	if err != nil {
		return nil
	}

	return nil
}
