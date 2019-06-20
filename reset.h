#ifndef RESET_H
#define RESET_H

#include <pcap.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <signal.h>
#include <unistd.h> 

#define SNAP_LEN 1518

#define ETHHDRSIZE 14

#define ETHER_ADDR_LEN  6

#define IPHDRSIZE sizeof(struct sniff_ip)
#define TCPHDRSIZE sizeof(struct sniff_tcp)
#define IPTCPHDRSIZE IPHDRSIZE + TCPHDRSIZE

#define HOME_IP ""

#define CAPTURE_COUNT -1         

extern FILE* logfile;

struct sniff_ethernet {
  u_char  ether_dhost[ETHER_ADDR_LEN];    /* destination host address */
  u_char  ether_shost[ETHER_ADDR_LEN];    /* source host address */
  u_short ether_type;                     /* IP? ARP? RARP? etc */
};

struct sniff_ip {
  u_char  ip_vhl;               /* version << 4 | header length >> 2 */
  u_char  ip_tos;               /* type of service */
  u_short ip_len;               /* total length */
  u_short ip_id;                /* identification */
  u_short ip_off;               /* fragment offset field */
#define IP_RF 0x8000            /* reserved fragment flag */
#define IP_DF 0x4000            /* dont fragment flag */
#define IP_MF 0x2000            /* more fragments flag */
#define IP_OFFMASK 0x1fff       /* mask for fragmenting bits */
  u_char  ip_ttl;               /* time to live */
  u_char  ip_p;                 /* protocol */
  u_short ip_sum;               /* checksum */
  struct  in_addr ip_src,ip_dst;/* source and dest address */
};
#define IP_HL(ip)               (((ip)->ip_vhl) & 0x0f)
#define IP_V(ip)                (((ip)->ip_vhl) >> 4)

typedef u_int tcp_seq;

struct sniff_tcp {
  u_short th_sport;       /* source port */
  u_short th_dport;       /* destination port */
  tcp_seq th_seq;         /* sequence number */
  tcp_seq th_ack;         /* acknowledgement number */
  u_char  th_offx2;       /* data offset, rsvd */
#define TH_OFF(th)      (((th)->th_offx2 & 0xf0) >> 4)
  u_char  th_flags;
#define TH_FIN  0x01    /* 1  */
#define TH_SYN  0x02    /* 2  */
#define TH_RST  0x04    /* 4  */
#define TH_PUSH 0x08    /* 8  */
#define TH_ACK  0x10    /* 16 */
#define TH_URG  0x20    /* 32 */
#define TH_ECE  0x40    /* 64 */
#define TH_CWR  0x80    /* 128*/
#define TH_FLAGS        (TH_FIN|TH_SYN|TH_RST|TH_ACK|TH_URG|TH_ECE|TH_CWR)
  u_short th_win;         /* window */
  u_short th_sum;         /* checksum */
  u_short th_urp;         /* urgent pointer */
};

struct pseudo_hdr {
  u_int32_t src;     /* 32bit source ip address*/
  u_int32_t dst;     /* 32bit destination ip address */  
  u_char zero;       /* 8 reserved bits (all 0)  */
  u_char protocol;   /* protocol field of ip header */
  u_int16_t tcplen;  /* tcp length (both header and data */
};


void got_packet(const u_char *packet);
void ParseTCPPacket(const u_char *packet);
void createRSTpacket(struct in_addr srcip, struct in_addr destip, u_short sport, u_short dport,
                     u_short ident, unsigned int seq, u_char ttl, unsigned int ack);

unsigned short in_cksum(unsigned short *addr,int len){
  register int sum = 0;
  u_short answer = 0;
  register u_short *w = addr;
  register int nleft = len;

  while (nleft > 1) {
    sum += *w++;
    nleft -= 2;
  }
  
  
  if (nleft == 1) {
    *(u_char *)(&answer) = *(u_char *) w;
    sum += answer;
  }
  sum = (sum >> 16) + (sum &0xffff); /* add hi 16 to low 16 */
  sum += (sum >> 16); /* add carry */
  answer = ~sum; /* truncate to 16 bits */
  return(answer);
  
}
void got_packet(const u_char *packet)
{
  const struct sniff_ip *ip;              /* The IP header */
  int size_ip;
  char srcHost[INET_ADDRSTRLEN];
  char dstHost[INET_ADDRSTRLEN];
  
  /* define/compute ip header offset */
  ip = (struct sniff_ip*)(packet + ETHHDRSIZE);

  size_ip = IP_HL(ip)*4;
  if (size_ip < IPHDRSIZE) {
    printf("   * Invalid IP header length: %u bytes\n", size_ip);
    return;
  }
  
  strcpy(srcHost, inet_ntoa(ip->ip_src));
  strcpy(dstHost, inet_ntoa(ip->ip_dst));
  fprintf(logfile, "%s\t->\t", srcHost);
  fprintf(logfile, " %s\n", dstHost);
  
  switch(ip->ip_p) {
    case IPPROTO_TCP:
      //printf("TCP\t");
      ParseTCPPacket((u_char *)ip);
      break;/*
    case IPPROTO_UDP:
      printf("UDP\t");
      break;
    case IPPROTO_ICMP:
      printf("ICMP\t");
      break;
    case IPPROTO_IP:
      printf("IP\t");
      break;*/
    default:
      //printf("Protocol: unknown\t");
      break;
  }
}

void ParseTCPPacket(const u_char *packet)
{
    const struct sniff_ip *ip;              /* The IP header */
    const struct sniff_tcp *tcp;            /* The TCP header */
    //const u_char *payload;                  /* Packet payload */
    
    //int size_ip;
    int size_tcp;
    //int size_payload;

    unsigned int srcport;
    unsigned int dstport;
    
    ip = (struct sniff_ip*)(packet);
    
    {
      /* define/compute tcp header offset */
      tcp = (struct sniff_tcp*)(packet + IPHDRSIZE);
      size_tcp = TH_OFF(tcp)*4;
      if (size_tcp < TCPHDRSIZE) {
        printf("   * Invalid TCP header length: %u bytes\n", size_tcp);
        return;
      }
      srcport = ntohs(tcp->th_sport);
      dstport = ntohs(tcp->th_dport);
      
      /* define/compute tcp payload (segment) offset
	// This is the payload in which u have to search for signatures
      payload = (u_char *)(packet + IPHDRSIZE + TCPHDRSIZE);
      
      /* compute tcp payload (segment) size 
      size_payload = ntohs(ip->ip_len) - (size_ip + size_tcp);
      */
      fprintf(logfile, "%d\t->\t", srcport);
      fprintf(logfile, " %d\n", dstport); 
      
      createRSTpacket(ip->ip_dst, ip->ip_src, tcp->th_dport, tcp->th_sport, ip->ip_id, tcp->th_ack, ip->ip_ttl, tcp->th_ack);
      createRSTpacket(ip->ip_src, ip->ip_dst, tcp->th_sport, tcp->th_dport, ip->ip_id, tcp->th_ack, ip->ip_ttl, tcp->th_ack);
    }
}

void createRSTpacket(struct  in_addr srcip, struct  in_addr destip, u_short sport, u_short dport, u_short ident, unsigned int seq, u_char  ttl, unsigned int ack) {
	//printf("in CreateRstPacket\n");
  int sockfd;
  struct sockaddr_in dstaddr;
  char datagram[4096];  /* buffer for datagrams */
  struct sniff_ip *iph = (struct sniff_ip *) datagram;
  struct sniff_tcp *tcph = (struct sniff_tcp *) (datagram + sizeof (struct sniff_ip));
  int one = 1;
  const int *val = &one;
  struct pseudo_hdr *phdr;
  char temp_addr[INET_ADDRSTRLEN];
  
  
  if ((sockfd = socket(AF_INET, SOCK_RAW, IPPROTO_TCP)) < 0) {
    perror("createRSTpacket() sock failed:");
    exit(EXIT_FAILURE);
  }
  /* Recommended by Stevens: you need the "one" variable for setsockopt
   call so here it is*/
  if (setsockopt (sockfd, IPPROTO_IP, IP_HDRINCL, val, sizeof (one)) < 0){
    printf ("Warning: Cannot set HDRINCL from port %d to port %d\n", 
            ntohs(sport), ntohs(dport));
    perror("setsockopt: ");
  }
  
  strncpy(temp_addr, inet_ntoa(destip), INET_ADDRSTRLEN); /*BUG: destip or srcip?*/
  dstaddr.sin_family = AF_INET;
  dstaddr.sin_port = dport;
  inet_pton(AF_INET, temp_addr, &dstaddr.sin_addr);
  
  memset (datagram, 0, 4096);       /* zero out the buffer */
  iph->ip_vhl = 0x45;               /* version=4,header_length=5 */
  iph->ip_tos = 0;                  /* type of service not needed */
  /* 
   wierd thing [TODO][BUG]:
   htons() for linux
   no htons for mac os x/BSD 
   */
  iph->ip_len = htons(IPTCPHDRSIZE);     /* no payload for RST */
  iph->ip_id  = ident;              /* ID could this be random?*/
  iph->ip_off = 0;                  /* no fragmentation */
  iph->ip_ttl = ttl;                /* Time to Live, default:255 */
  iph->ip_src = srcip;              /* faking source device IP */
  iph->ip_dst = destip;             /* target destination address */
  iph->ip_sum = 0;                  /* Checksum is set to zero until computed */
  iph->ip_p   = IPPROTO_TCP;        /* IPPROTO_TCP or IPPROTO_UDP */
  iph->ip_sum = in_cksum((unsigned short *)iph, IPHDRSIZE); 

  tcph->th_sport = sport;              /* faking source port */
  tcph->th_dport = dport;              /* target destination port */
  tcph->th_seq   = seq;                /* SYN sequence the should be 
                                        incremented in one dir and
                                        echoed in the other */
  tcph->th_ack   = 0;                  /* No ACK needed? or echo ACK?*/
  tcph->th_offx2 = 0x50;               /* 50h (5 offset) ( 8 0s reserved )*/
  tcph->th_flags = TH_RST;             /* initial connection request FLAG*/
  tcph->th_win   =  0;                 /* Window size default: htons(4500) + rand()%1000  */
                                       /* maximum allowed window size 65k*/
  tcph->th_urp   = 0;                  /* no urgent pointer */
  tcph->th_sum=0;                      /* Checksum is set to zero until computed */

  /* pseudo header for tcp checksum */
  phdr = (struct pseudo_hdr *) (datagram + IPTCPHDRSIZE);
  phdr->src = srcip.s_addr;
  phdr->dst = destip.s_addr;
  phdr->zero = 0;
  phdr->protocol = IPPROTO_TCP;
  phdr->tcplen = htons(TCPHDRSIZE);       
  /* in bytes the tcp segment length default:0x14*/
  tcph->th_sum = in_cksum((unsigned short *)(tcph), IPTCPHDRSIZE);
  
  /*printf("RST packet IP Header\n");*/
 
  if (sendto(sockfd, datagram, IPTCPHDRSIZE, 0, (struct sockaddr *)&dstaddr, sizeof(dstaddr)) < 0) {
    fprintf(stderr, "Error sending datagram: from port %d to port %d\n", 
            ntohs(sport), ntohs(dport));
    perror("sendto: ");
  }
  else {
    //printf("verify: %s %d\n", inet_ntoa(dstaddr.sin_addr), ntohs(dstaddr.sin_port));
    
  }
  
  close(sockfd);
}



#endif
