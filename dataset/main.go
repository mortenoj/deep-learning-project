package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/mortenoj/deep-learning-project/dataset/webscraper"
	"github.com/sirupsen/logrus"

	dem "github.com/markus-wa/demoinfocs-golang"
	"github.com/markus-wa/demoinfocs-golang/common"
	events "github.com/markus-wa/demoinfocs-golang/events"
	"github.com/mortenoj/deep-learning-project/dataset/inventory"
	"github.com/mortenoj/deep-learning-project/dataset/utils"
)

// Run like this: go run print_kills.go
func main() {
	//var err error
	inventory.Init()

	var (
		wg    sync.WaitGroup
		links []string
		err   error
	)

	matches, err := filepath.Glob("demos/*")
	for _, file := range matches {
		os.RemoveAll(file)
	}

	errChannel := make(chan error, 1)
	finished := make(chan bool, 1)

	for i := 0; i < 2; i++ {
		l, err := webscraper.GetLinks(40*i, 40)
		if err != nil {
			panic(err)
		}

		links = append(links, l...)

		time.Sleep(time.Second * 10)
	}

	logrus.Info("Available links: ", len(links))

	chunks := utils.ChunkStrings(links, 2)

	wg.Add(len(links))
	for i, links := range chunks {
		go func(links []string, wg *sync.WaitGroup, key int) {
			err = handleDemoLinks(links, key+1)
			if err != nil {
				panic(err)
			}
		}(links, &wg, i)
	}

	go func() {
		wg.Wait()
		close(finished)
	}()

	select {
	case <-finished:
	case err := <-errChannel:
		if err != nil {
			panic(err)
		}
	}

	// Debug TODO: Remove
	//extractDataset("demos/fnatic-vs-vitality-m1-nuke.dem")
}

func handleDemoLinks(links []string, key int) error {
	for _, demoLink := range links {
		//rarPath := "demos/demo.rar"
		rarPath := fmt.Sprintf("demos/%d/demos.rar", key)
		os.MkdirAll(fmt.Sprintf("demos/%d", key), os.ModePerm)

		err := utils.DownloadDemo(demoLink, rarPath)
		if err != nil {
			return err
		}

		files, err := utils.UnRar(rarPath, fmt.Sprintf("demos/%d", key))
		if err != nil {
			return err
		}

		for _, file := range files {
			err = extractDataset(file)
			if err != nil {
				return err
			}
		}

		matches, err := filepath.Glob(fmt.Sprintf("demos/%d/*", key))
		for _, file := range matches {
			os.RemoveAll(file)
		}

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

	var gameInfo []inventory.RoundInfo
	var roundInfo inventory.RoundInfo

	tScoreRoundStart := 0

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

		ctPlayers := ct.Members()
		tPlayers := t.Members()

		ctEquipment := inventory.GetEquipmentList()
		tEquipment := inventory.GetEquipmentList()

		for _, player := range ctPlayers {
			for _, weapon := range player.Weapons() {
				ctEquipment = inventory.AddToList(ctEquipment, weapon.Weapon)
			}
			if player.HasDefuseKit {
				ctEquipment = inventory.AddToList(ctEquipment, common.EqDefuseKit)
			}
			if player.Armor > 0 {
				ctEquipment = inventory.AddToList(ctEquipment, common.EqKevlar)
			}
			if player.HasHelmet {
				ctEquipment = inventory.AddToList(ctEquipment, common.EqHelmet)
			}
		}

		for _, player := range tPlayers {
			for _, weapon := range player.Weapons() {
				tEquipment = inventory.AddToList(tEquipment, weapon.Weapon)
			}
			if player.Armor > 0 {
				tEquipment = inventory.AddToList(tEquipment, common.EqKevlar)
			}
			if player.HasHelmet {
				tEquipment = inventory.AddToList(tEquipment, common.EqHelmet)
			}
		}

		roundInfo.CTEquipment = ctEquipment
		roundInfo.CTSideTotalWorth = ct.CurrentEquipmentValue()
		roundInfo.TEquipment = tEquipment
		roundInfo.TSideTotalWorth = t.CurrentEquipmentValue()
	})

	// RoundEndOffical is the most accurate and gives what we want, but it does not give the result of the last round
	// However we can know who won by the team that had the highest Score last
	p.RegisterEventHandler(func(e events.RoundEndOfficial) {
		gs := p.GameState()

		if gs.IsMatchStarted() != true {
			return
		}
		t := gs.TeamTerrorists()

		if t.Score > tScoreRoundStart {
			roundInfo.TerroristsWon = true
		} else {
			roundInfo.TerroristsWon = false
		}

		roundInfo.Map = p.Header().MapName

		gameInfo = append(gameInfo, roundInfo)

		// Save round winner in a array or something e.g. 1 = T win, 0 = CT win
		// Save these to file when everything is done parsing
	})

	// Parse to end
	err = p.ParseToEnd()
	if err != nil {
		return nil
	}

	// Gathering the round winner of the last round as it is not collected in events.RoundEndOfficial
	gs := p.GameState()
	ct := gs.TeamCounterTerrorists()
	t := gs.TeamTerrorists()
	if t.Score > ct.Score {
		roundInfo.TerroristsWon = true
	} else {
		roundInfo.TerroristsWon = false
	}

	roundInfo.Map = p.Header().MapName
	gameInfo = append(gameInfo, roundInfo)

	// TODO: Caused some errors, need to be looked into, not vital though
	// Writing to file with a unique name identifier of the demo name
	//fileName := strings.Split(filePath, "/")
	inventory.WriteToJSON(gameInfo, "test")
	return nil
}
