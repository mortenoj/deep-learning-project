package inventory

import (
	"encoding/json"
	"fmt"
	"github.com/markus-wa/demoinfocs-golang/common"
	"io/ioutil"
	"strconv"
)

type Equipment struct{
	Name string
	Count int
}

// TODO: We could possibly also have info like what map etc
type RoundInfo struct{
	CTEquipment []Equipment
	CTSideTotalWorth int
	TEquipment []Equipment
	TSideTotalWorth int
	TerroristsWon bool
}

var equipmentCount int
var resultFiles int
var equipmentList []Equipment
var equipmentKeys map[common.EquipmentElement]int // To be able to easier insert weapons correctly when retrieving weapons

//TODO: Make sure we have same array structure for image output as for this output,
// in regards to making the array indices the same for every equipment type
// Maps seem to have random order? Maybe need to hardcode this instead
func Init(){
	resultFiles = 0
	initEquipment()
}

func GetEquipmentList() []Equipment{
	returnList := make([]Equipment, len(equipmentList))
	copy(returnList, equipmentList)

	return returnList
}

func WriteToJSON(gameInfo []RoundInfo, fileName string){
	file, _ := json.MarshalIndent(gameInfo, "", " ")
	resultFiles += 1
	filePath := "output/results_" + strconv.Itoa(resultFiles) + "_" + fileName + ".json"
	_ = ioutil.WriteFile(filePath, file, 0644)

	fmt.Printf("Wrote output to " + filePath + "\n")
}

func AddToList(list []Equipment, e common.EquipmentElement) []Equipment{
	list[equipmentKeys[e]].Count += 1
	return list
}

// Using a hardcoded array because we don't want the data to lie randomly everytime like it does with a map

func initEquipment(){
	equipmentKeys = make(map[common.EquipmentElement]int)
	AddEquipment("AK-47", common.EqAK47)
	AddEquipment("AUG", common.EqAUG)
	AddEquipment("AWP", common.EqAWP)
	AddEquipment("PP-Bizon", common.EqBizon)
	AddEquipment("C4", common.EqBomb)
	AddEquipment("Desert Eagle", common.EqDeagle)
	AddEquipment("Decoy Grenade", common.EqDecoy)
	AddEquipment("Dual Berettas", common.EqDualBerettas)
	AddEquipment("FAMAS", common.EqFamas)
	AddEquipment("Five-SeveN", common.EqFiveSeven)
	AddEquipment("Flashbang", common.EqFlash)
	AddEquipment("G3SG1", common.EqG3SG1)
	AddEquipment("Galil AR", common.EqGalil)
	AddEquipment("Glock-18", common.EqGlock)
	AddEquipment("HE Grenade", common.EqHE)
	AddEquipment("P2000", common.EqP2000)
	AddEquipment("Incendiary Grenade", common.EqIncendiary)
	AddEquipment("M249", common.EqM249)
	AddEquipment("M4A4", common.EqM4A4)
	AddEquipment("MAC-10", common.EqMac10)
	AddEquipment("MAG-7", common.EqSwag7)
	AddEquipment("Molotov", common.EqMolotov)
	AddEquipment("MP7", common.EqMP7)
	AddEquipment("MP5-SD", common.EqMP5)
	AddEquipment("MP9", common.EqMP9)
	AddEquipment("Negev", common.EqNegev)
	AddEquipment("Nova", common.EqNova)
	AddEquipment("p250", common.EqP250)
	AddEquipment("P90", common.EqP90)
	AddEquipment("Sawed-Off", common.EqSawedOff)
	AddEquipment("SCAR-20", common.EqScar20)
	AddEquipment("SG 553", common.EqSG553)
	AddEquipment("Smoke Grenade", common.EqSmoke)
	AddEquipment("SSG 08", common.EqScout)
	AddEquipment("Zeus x27", common.EqZeus)
	AddEquipment("Tec-9", common.EqTec9)
	AddEquipment("UMP-45", common.EqUMP)
	AddEquipment("XM1014", common.EqXM1014)
	AddEquipment("M4A1", common.EqM4A1)
	AddEquipment("CZ75 Auto", common.EqCZ)
	AddEquipment("USP-S", common.EqUSP)
	AddEquipment("World", common.EqWorld)
	AddEquipment("R8 Revolver", common.EqRevolver)
	AddEquipment("Kevlar Vest", common.EqKevlar)
	AddEquipment("Kevlar + Helmet", common.EqHelmet)
	AddEquipment("Defuse Kit", common.EqDefuseKit)
	AddEquipment("Knife", common.EqKnife)
	AddEquipment("UNKNOWN", common.EqUnknown)
}

func AddEquipment(name string, elem common.EquipmentElement){
	newEquipment := Equipment{ Name: name, Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	equipmentKeys[elem] = equipmentCount
	equipmentCount += 1
}


