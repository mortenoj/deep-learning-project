package main

import (
	"fmt"
	"os"

	dem "github.com/markus-wa/demoinfocs-golang"
	events "github.com/markus-wa/demoinfocs-golang/events"
)

// Run like this: go run print_kills.go
func main() {
	f, err := os.Open("demos/north-vs-aristocracy-m1-nuke.dem")
	if err != nil {
		panic(err)
	}
	defer f.Close()

	p := dem.NewParser(f)

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

		serverPlayers := ct.Members()
		for _, p := range t.Members() {
			serverPlayers = append(serverPlayers, p)
		}

		for _, player := range serverPlayers {
			fmt.Printf("%v \n", player.Weapons())
			fmt.Printf("%v \n", player.RawWeapons)
		}

		//ctEquipment := ct.FreezeTimeEndEquipmentValue()
		//tEquipment := t.FreezeTimeEndEquipmentValue()

		//fmt.Printf("CT Equipment value: %d\n", ctEquipment)
		//fmt.Printf("CT Equipment value: %d\n", ct.CurrentEquipmentValue())
		//fmt.Printf("T Equipment value: %d\n", tEquipment)
		//fmt.Printf("T Equipment value: %d\n", t.CurrentEquipmentValue())
		fmt.Printf("CT equipment value: %d\n", t.CurrentEquipmentValue())
	})

	// Parse to end
	err = p.ParseToEnd()
	if err != nil {
		panic(err)
	}
}
